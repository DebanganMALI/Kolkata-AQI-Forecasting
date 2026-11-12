"""
Preprocessing: turn the raw Kolkata rows into a clean daily table.

Rules followed here (see reports/preprocessing_decisions.md):
  * The raw file is never modified; the result is saved to data/processed/.
  * Only dates from config.START_DATE onwards are used.
  * Xylene is dropped (too incomplete).
  * Any remaining gap in an INPUT column is filled with the previous day's
    value (forward fill). This uses only the past, never the future. Every
    filled cell is marked in a "<column>_imputed" flag column, so real
    observations and filled values can always be told apart.
  * The TARGET (AQI) is never filled. If AQI were missing inside the chosen
    period the script stops with an error instead of inventing values.
  * The scaler (mean / standard deviation) is learned from the TRAINING
    period only, then applied unchanged to validation and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_loader import load_city, load_raw


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
def feature_columns() -> list[str]:
    """Model input columns, in a fixed order. AQI itself (past values) is the first input."""
    cols = [config.TARGET] + list(config.POLLUTANT_FEATURES)
    if config.USE_CALENDAR_FEATURES:
        cols += ["doy_sin", "doy_cos"]
    return cols


def assign_split(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Label each date train / val / test using the fixed cut-off dates."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    train_end = np.datetime64(pd.Timestamp(config.TRAIN_END))
    val_end = np.datetime64(pd.Timestamp(config.VAL_END))
    return np.where(d <= train_end, "train", np.where(d <= val_end, "val", "test"))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def build_clean_series(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean daily table, log of what was done)."""
    log: dict = {}
    city = load_city(raw)
    log["city_rows_total"] = len(city)

    start, end = pd.Timestamp(config.START_DATE), pd.Timestamp(config.END_DATE)
    city = city[(city["Date"] >= start) & (city["Date"] <= end)]

    # Re-index to a full calendar so any absent date would show up as a gap.
    calendar = pd.date_range(start, end, freq="D")
    keep = [config.TARGET] + list(config.POLLUTANT_FEATURES)
    df = city.set_index("Date")[keep].reindex(calendar)
    df.index.name = "Date"
    log["calendar_days"] = len(calendar)
    log["rows_present"] = int(city.shape[0])

    # The target must be real everywhere.
    n_missing_target = int(df[config.TARGET].isna().sum())
    if n_missing_target:
        missing_days = df.index[df[config.TARGET].isna()].date.tolist()
        raise ValueError(
            f"AQI is missing on {n_missing_target} day(s) inside the modelling period "
            f"({missing_days[:10]}...). Refusing to invent target values - "
            "adjust START_DATE/END_DATE in src/config.py."
        )

    # Forward-fill input gaps (past-only) and flag every filled cell.
    filled_log = {}
    for col in config.POLLUTANT_FEATURES:
        missing = df[col].isna()
        df[f"{col}_imputed"] = missing
        if missing.any():
            filled_log[col] = [d.date().isoformat() for d in df.index[missing]]
            df[col] = df[col].ffill()
            if df[col].isna().any():
                raise ValueError(
                    f"{col} is missing at the very start of the period, so there is "
                    "no past value to carry forward."
                )
    df[f"{config.TARGET}_imputed"] = False
    log["forward_filled"] = filled_log
    log["n_cells_filled"] = sum(len(v) for v in filled_log.values())

    # Calendar features: position in the year as a point on a circle, so
    # 31 December and 1 January are close together.
    doy = df.index.day_of_year.to_numpy()
    days_in_year = np.where(df.index.is_leap_year, 366, 365)
    angle = 2 * np.pi * (doy - 1) / days_in_year
    df["doy_sin"] = np.sin(angle)
    df["doy_cos"] = np.cos(angle)

    df["split"] = assign_split(df.index)
    df = df.reset_index()
    log["split_counts"] = df["split"].value_counts().reindex(["train", "val", "test"]).to_dict()
    log["split_high_days"] = (
        df.assign(high=df[config.TARGET] > config.ALERT_THRESHOLD)
        .groupby("split")["high"].sum().reindex(["train", "val", "test"]).astype(int).to_dict()
    )
    return df, log


# ---------------------------------------------------------------------------
# Scaling (fit on training data only)
# ---------------------------------------------------------------------------
@dataclass
class StandardScaler:
    """z = (x - mean) / std, with mean and std learned from training rows only."""

    columns: list[str]
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "StandardScaler":
        values = df[columns].to_numpy(dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0  # a constant column would otherwise divide by zero
        return cls(list(columns), mean.tolist(), std.tolist())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=float)
        return (x - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse_column(self, z: np.ndarray, column: str) -> np.ndarray:
        i = self.columns.index(column)
        return np.asarray(z) * self.std[i] + self.mean[i]

    def scale_column(self, x: np.ndarray, column: str) -> np.ndarray:
        i = self.columns.index(column)
        return (np.asarray(x) - self.mean[i]) / self.std[i]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "StandardScaler":
        return cls(**json.loads(Path(path).read_text()))


def fit_scaler(clean: pd.DataFrame) -> StandardScaler:
    return StandardScaler.fit(clean[clean["split"] == "train"], feature_columns())


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------
def write_decisions(log: dict, clean: pd.DataFrame) -> None:
    by_split = clean.groupby("split")["Date"].agg(["min", "max", "count"]).reindex(["train", "val", "test"])
    L = [
        "# Preprocessing Decisions",
        "",
        f"*Generated by `python main.py preprocess` on {datetime.now():%Y-%m-%d %H:%M}. "
        "Numbers are computed from the data at run time.*",
        "",
        "| # | Decision | Reason (evidence from `reports/dataset_audit.md`) |",
        "|---|---|---|",
        f"| 1 | Use only Kolkata, {config.START_DATE} to {config.END_DATE} "
        f"({log['calendar_days']} days) | Before {config.START_DATE}: AQI missing 59 days in a row "
        "(2018-04-18 to 2018-06-15); CO = 0 for 47 days and Toluene = 0 for 8 days (likely "
        "'no reading' stored as 0). After it, AQI is unbroken. |",
        "| 2 | Raw file never modified | Cleaned table saved separately to "
        "`data/processed/kolkata_daily_clean.csv`. |",
        "| 3 | Drop Xylene | 51.6 % missing for Kolkata, only recorded from mid-2019. |",
        f"| 4 | Inputs: past AQI + {len(config.POLLUTANT_FEATURES)} pollutants"
        + (" + day-of-year (sin/cos)" if config.USE_CALENDAR_FEATURES else "")
        + " | All available in the CSV. No weather or external data added. Day-of-year is "
        "known in advance, so it is not future information. |",
        f"| 5 | AQI (target) never filled | AQI missing days inside the period: 0. The code "
        "stops with an error if this ever changes. |",
        f"| 6 | Input gaps forward-filled, flagged | Cells filled: {log['n_cells_filled']} "
        f"({', '.join(f'{k}: {v}' for k, v in log['forward_filled'].items()) or 'none'}). "
        "Forward fill copies the previous day's value, so it uses only the past. Each filled "
        "cell is marked in a `<column>_imputed` column. |",
        f"| 7 | Chronological split by forecast date | Train {by_split.loc['train','min'].date()} to "
        f"{by_split.loc['train','max'].date()} ({by_split.loc['train','count']} days, "
        f"{log['split_high_days']['train']} with AQI > {config.ALERT_THRESHOLD}); validation "
        f"{by_split.loc['val','min'].date()} to {by_split.loc['val','max'].date()} "
        f"({by_split.loc['val','count']} days, {log['split_high_days']['val']} high); test "
        f"{by_split.loc['test','min'].date()} to {by_split.loc['test','max'].date()} "
        f"({by_split.loc['test','count']} days, {log['split_high_days']['test']} high). "
        "A plain 70/15/15 split left only 2 high days in test, so the alert could not be "
        "evaluated. Cut-offs were fixed from the AQI distribution before any model was trained. |",
        "| 8 | No random shuffling across time | Splits are blocks of consecutive dates; "
        "training batches may be shuffled *within* the training period only. |",
        "| 9 | Scaler fitted on training period only | Mean / std from train rows; the same "
        "numbers are reused for validation and test (`data/processed/scaler.json`). |",
        f"| 10 | Lookback {config.LOOKBACK} days, horizon {config.HORIZON} day | Each sample uses "
        f"days t-{config.LOOKBACK - 1}..t to predict AQI on day t+1. A validation/test sample may "
        "use history from the end of the previous period (that is still the past, so it is "
        "allowed); its target is always inside its own period. |",
        f"| 11 | Alert rule: predicted AQI > {config.ALERT_THRESHOLD} | Project requirement "
        "(CPCB 'Poor' category starts at 201). |",
        "",
        "Known limitations carried forward: about 2 years of data (two winters); the test "
        "period includes the COVID-19 lockdown (from 2020-03-25), which changed pollution "
        "patterns; no weather data.",
        "",
    ]
    config.DECISIONS_MD.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
def run_preprocessing(verbose: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    raw = load_raw()
    clean, log = build_clean_series(raw)
    scaler = fit_scaler(clean)

    config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = clean.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(config.PROCESSED_CSV, index=False)
    scaler.to_json(config.SCALER_JSON)
    write_decisions(log, clean)

    if verbose:
        print(f"Clean table : {config.PROCESSED_CSV} ({len(clean)} days)")
        print(f"Split days  : {log['split_counts']}")
        print(f"AQI > {config.ALERT_THRESHOLD} days per split: {log['split_high_days']}")
        print(f"Cells forward-filled: {log['n_cells_filled']} {log['forward_filled']}")
        print(f"Scaler      : {config.SCALER_JSON} (fitted on train only)")
        print(f"Decisions   : {config.DECISIONS_MD}")
    return clean, scaler


def load_processed() -> pd.DataFrame:
    if not config.PROCESSED_CSV.exists():
        raise FileNotFoundError("Run `python main.py preprocess` first.")
    df = pd.read_csv(config.PROCESSED_CSV, parse_dates=["Date"])
    return df


if __name__ == "__main__":
    run_preprocessing()
