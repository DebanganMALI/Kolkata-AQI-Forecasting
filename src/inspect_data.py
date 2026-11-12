"""
Dataset inspection and audit for the Kolkata Air Quality Forecasting project.

What this script does
---------------------
1. Reads the raw Kaggle file  data/raw/city_day.csv  (read-only - it is never
   modified, cleaned, or re-saved).
2. Keeps only the rows where City == "Kolkata".
3. Computes factual statistics: row counts, date coverage, missing values,
   duplicates, missing dates, AQI distribution, implausible values, and how
   many usable next-day forecasting samples exist WITHOUT any imputation.
4. Writes a Markdown report to  reports/dataset_audit.md  and (if matplotlib is
   installed) three-four figures to  reports/figures/.

What this script deliberately does NOT do
-----------------------------------------
- No imputation (filling gaps), no outlier removal, no feature engineering.
- No model training.
- No external / weather data.
Every number in the report is calculated from the CSV at run time.

Run from the project root:
    python src/inspect_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so it works from the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "city_day.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_audit.md"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

CITY = "Kolkata"
ALERT_THRESHOLD = 200  # project's high-pollution alert level

# Indian National AQI (CPCB) categories. Upper bound is inclusive.
AQI_CATEGORIES = [
    ("Good", 0, 50),
    ("Satisfactory", 51, 100),
    ("Moderate", 101, 200),
    ("Poor", 201, 300),
    ("Very Poor", 301, 400),
    ("Severe", 401, 500),
]
CATEGORY_ORDER = [c[0] for c in AQI_CATEGORIES]

NON_POLLUTANT_COLUMNS = {"City", "Date", "AQI", "AQI_Bucket"}

# Lookback windows (days of history) used to count possible training samples.
LOOKBACKS = [1, 7, 14, 30]

# Illustrative chronological split (fractions of the calendar range).
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# A pollutant missing more than this share of days is called "excessively incomplete".
HEAVY_MISSING_PCT = 30.0

# Known external context (NOT derived from the CSV): India's national COVID-19
# lockdown began on 2020-03-25. Used only to count how many rows fall after it.
LOCKDOWN_START = pd.Timestamp("2020-03-25")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a Markdown table (no extra packages needed)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            elif v is None or (not isinstance(v, str) and pd.isna(v)):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aqi_to_category(aqi: float) -> str | None:
    """Map an AQI number to its CPCB category (float-safe boundaries)."""
    if pd.isna(aqi):
        return None
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def runs_of_true(mask: pd.Series) -> pd.DataFrame:
    """Return start/end/length of each consecutive run where mask is True."""
    m = mask.to_numpy(dtype=bool)
    idx = mask.index
    out = []
    start = None
    for i, val in enumerate(m):
        if val and start is None:
            start = i
        if (not val or i == len(m) - 1) and start is not None:
            end = i if val else i - 1
            out.append((idx[start], idx[end], end - start + 1))
            start = None
    return pd.DataFrame(out, columns=["start", "end", "days"])


def max_identical_run(s: pd.Series) -> tuple[int, object, object]:
    """Longest run of the same non-missing value on consecutive calendar days."""
    best, best_val, best_start = 0, None, None
    prev, run, run_start = None, 0, None
    for date, v in s.items():
        if pd.notna(v) and prev is not None and pd.notna(prev) and v == prev:
            run += 1
        else:
            run, run_start = (1, date) if pd.notna(v) else (0, None)
        if run > best:
            best, best_val, best_start = run, v, run_start
        prev = v
    return best, best_val, best_start


def count_windows(observed: pd.Series, lookback: int) -> pd.Series:
    """
    For each calendar day t, True if days t-lookback+1 .. t are all observed
    AND day t+1 (the target) is observed. Index = target date (t+1).
    No imputation: any missing day breaks the window.
    """
    obs = observed.astype(int)
    history_ok = obs.rolling(lookback, min_periods=lookback).sum() == lookback
    target_ok = observed.shift(-1, fill_value=False)
    valid = history_ok & target_ok
    # re-index to the target date
    valid.index = valid.index + pd.Timedelta(days=1)
    return valid


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------
def run_audit(input_path: Path, make_figures: bool = True) -> dict:
    if not input_path.exists():
        sys.exit(
            f"ERROR: {input_path} not found.\n"
            "Place the Kaggle file city_day.csv in data/raw/ and run again."
        )

    file_hash = sha256_of(input_path)
    raw = pd.read_csv(input_path)  # read only; never written back

    findings: dict = {"input": input_path, "sha256": file_hash}
    findings["all_rows"] = len(raw)
    findings["all_cities"] = raw["City"].nunique()
    findings["all_date_range"] = (raw["Date"].min(), raw["Date"].max())
    findings["columns"] = list(raw.columns)

    kol = raw[raw["City"] == CITY].copy()
    if kol.empty:
        sys.exit(f"ERROR: no rows with City == '{CITY}' found.")
    kol["Date"] = pd.to_datetime(kol["Date"], format="%Y-%m-%d")
    kol = kol.sort_values("Date").reset_index(drop=True)

    pollutants = [c for c in raw.columns if c not in NON_POLLUTANT_COLUMNS]
    numeric_cols = pollutants + ["AQI"]
    findings["pollutants"] = pollutants

    # ---- 1-2. counts and dates --------------------------------------------
    n = len(kol)
    first, last = kol["Date"].min(), kol["Date"].max()
    calendar = pd.date_range(first, last, freq="D")
    findings.update(n=n, first=first, last=last, calendar_days=len(calendar))

    # ---- 3-4. missing values ----------------------------------------------
    miss = pd.DataFrame(
        {
            "column": kol.columns,
            "missing_count": kol.isna().sum().to_numpy(),
        }
    )
    miss["missing_pct_of_rows"] = 100 * miss["missing_count"] / n
    miss["non_missing_count"] = n - miss["missing_count"]
    findings["missing"] = miss

    # Pollutants never measured at all / heavily incomplete
    pol_miss = miss[miss["column"].isin(pollutants)].set_index("column")
    findings["never_measured"] = pol_miss.index[pol_miss["missing_count"] == n].tolist()
    findings["heavy_missing"] = pol_miss.index[
        pol_miss["missing_pct_of_rows"] > HEAVY_MISSING_PCT
    ].tolist()

    # ---- 6. AQI distribution ----------------------------------------------
    aqi = kol["AQI"]
    findings["aqi_describe"] = aqi.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    bucket_counts = (
        kol["AQI_Bucket"].value_counts(dropna=False).rename_axis("AQI_Bucket")
    )
    bucket_tbl = pd.DataFrame(
        {
            "AQI_Bucket": CATEGORY_ORDER + ["(missing)"],
            "range": [f"{lo}-{hi}" for _, lo, hi in AQI_CATEGORIES] + [""],
        }
    )
    bucket_tbl["days"] = [
        int(bucket_counts.get(c, 0)) for c in CATEGORY_ORDER
    ] + [int(kol["AQI_Bucket"].isna().sum())]
    bucket_tbl["pct_of_rows"] = 100 * bucket_tbl["days"] / n
    findings["bucket_tbl"] = bucket_tbl
    findings["days_above_alert"] = int((aqi > ALERT_THRESHOLD).sum())

    # ---- 7. duplicates ----------------------------------------------------
    findings["dup_full_rows"] = int(kol.duplicated().sum())
    findings["dup_dates"] = int(kol["Date"].duplicated().sum())

    # ---- 8. missing dates and gaps ----------------------------------------
    daily = kol.set_index("Date").reindex(calendar)  # absent dates -> NaN rows
    row_absent = ~calendar.isin(kol["Date"])
    row_absent = pd.Series(row_absent, index=calendar)
    findings["absent_dates"] = int(row_absent.sum())
    findings["absent_runs"] = runs_of_true(row_absent)

    aqi_obs = daily["AQI"].notna()
    findings["aqi_observed_days"] = int(aqi_obs.sum())
    findings["aqi_missing_days_calendar"] = int((~aqi_obs).sum())
    aqi_gap_runs = runs_of_true(~aqi_obs)
    findings["aqi_gap_runs"] = aqi_gap_runs
    obs_runs = runs_of_true(aqi_obs)
    findings["aqi_obs_runs"] = obs_runs

    # Monthly coverage (calendar month x share of days with AQI observed)
    monthly = pd.DataFrame(
        {
            "calendar_days": aqi_obs.groupby(aqi_obs.index.to_period("M")).size(),
            "rows_present": (~row_absent).groupby(row_absent.index.to_period("M")).sum(),
            "aqi_observed": aqi_obs.groupby(aqi_obs.index.to_period("M")).sum(),
        }
    )
    monthly["aqi_coverage_pct"] = 100 * monthly["aqi_observed"] / monthly["calendar_days"]
    monthly["median_aqi"] = daily["AQI"].groupby(daily.index.to_period("M")).median()
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "month"
    findings["monthly"] = monthly

    # Per-column monthly coverage (for the heatmap)
    cov = (
        daily[numeric_cols].notna().groupby(daily.index.to_period("M")).mean() * 100
    )
    cov.index = cov.index.astype(str)
    findings["col_month_cov"] = cov

    # Calendar-month (seasonal) summary across all years
    by_moy = pd.DataFrame(
        {
            "aqi_observed_days": aqi_obs.groupby(aqi_obs.index.month).sum(),
            "median_aqi": daily["AQI"].groupby(daily.index.month).median(),
            "mean_aqi": daily["AQI"].groupby(daily.index.month).mean(),
            "days_aqi_gt_200": (daily["AQI"] > ALERT_THRESHOLD).groupby(daily.index.month).sum(),
        }
    )
    by_moy.index = [datetime(2000, m, 1).strftime("%b") for m in by_moy.index]
    by_moy.index.name = "month"
    findings["by_moy"] = by_moy

    # ---- anomalies / plausibility -----------------------------------------
    anom_rows = []
    for c in numeric_cols:
        s = kol[c].dropna()
        if s.empty:
            anom_rows.append((c, 0, 0, 0, np.nan, np.nan, np.nan, 0, ""))
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        extreme = int((s > q3 + 3 * iqr).sum())
        run_len, run_val, run_start = max_identical_run(daily[c])
        run_txt = (
            f"{run_len} days of {run_val:g} from {pd.Timestamp(run_start).date()}"
            if run_len >= 3
            else ""
        )
        anom_rows.append(
            (c, len(s), int((s < 0).sum()), int((s == 0).sum()),
             float(s.min()), float(s.median()), float(s.max()), extreme, run_txt)
        )
    anom = pd.DataFrame(
        anom_rows,
        columns=["column", "non_missing", "negative", "exact_zero", "min",
                 "median", "max", "far_outliers_(>Q3+3*IQR)", "longest_identical_run"],
    )
    findings["anom"] = anom

    findings["aqi_out_of_range"] = int(((aqi < 0) | (aqi > 500)).sum())
    both_pm = kol[["PM2.5", "PM10"]].dropna()
    findings["pm25_gt_pm10"] = int((both_pm["PM2.5"] > both_pm["PM10"]).sum())
    findings["pm_both_present"] = len(both_pm)
    findings["pm25_gt_pm10_dates"] = (
        kol.loc[both_pm.index[both_pm["PM2.5"] > both_pm["PM10"]], "Date"].dt.date.tolist()
    )

    expected_bucket = kol["AQI"].map(aqi_to_category)
    has_both = kol["AQI"].notna() & kol["AQI_Bucket"].notna()
    findings["bucket_mismatch"] = int((expected_bucket[has_both] != kol.loc[has_both, "AQI_Bucket"]).sum())
    findings["aqi_without_bucket"] = int((kol["AQI"].notna() & kol["AQI_Bucket"].isna()).sum())
    findings["bucket_without_aqi"] = int((kol["AQI"].isna() & kol["AQI_Bucket"].notna()).sum())
    findings["top_aqi_days"] = (
        kol.nlargest(5, "AQI")[["Date", "AQI", "AQI_Bucket", "PM2.5", "PM10"]]
        .assign(Date=lambda d: d["Date"].dt.date)
    )
    findings["bottom_aqi_days"] = (
        kol.nsmallest(5, "AQI")[["Date", "AQI", "AQI_Bucket", "PM2.5", "PM10"]]
        .assign(Date=lambda d: d["Date"].dt.date)
    )

    # Day-to-day AQI change (only between consecutive observed days)
    diff = daily["AQI"].diff()
    findings["aqi_daily_change"] = diff.abs().describe(percentiles=[0.5, 0.9, 0.99])
    jumps = pd.DataFrame({"date": daily.index.date, "previous_day_AQI": daily["AQI"].shift(1),
                          "AQI": daily["AQI"], "change": diff})
    findings["top_jumps"] = jumps.dropna().reindex(
        jumps["change"].abs().dropna().sort_values(ascending=False).index[:5])

    # ---- 9. next-day forecasting samples (no imputation) -------------------
    usable_pollutants = [
        p for p in pollutants
        if pol_miss.loc[p, "missing_pct_of_rows"] <= HEAVY_MISSING_PCT
    ]
    findings["usable_pollutants"] = usable_pollutants
    multi_obs = daily[["AQI"] + usable_pollutants].notna().all(axis=1)
    findings["multi_observed_days"] = int(multi_obs.sum())

    win_rows, windows = [], {}
    for L in LOOKBACKS:
        v_aqi = count_windows(aqi_obs, L)
        v_multi = count_windows(multi_obs, L)
        windows[L] = v_aqi
        win_rows.append((L, int(v_aqi.sum()), int(v_multi.sum())))
    findings["windows"] = pd.DataFrame(
        win_rows,
        columns=["lookback_days", "samples_AQI_only",
                 "samples_AQI_+_usable_pollutants"],
    )

    # ---- 10. illustrative chronological split -----------------------------
    total_days = len(calendar)
    cut1 = calendar[int(total_days * SPLIT_FRACTIONS[0])]
    cut2 = calendar[int(total_days * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))]
    bounds = [("train", first, cut1 - pd.Timedelta(days=1)),
              ("validation", cut1, cut2 - pd.Timedelta(days=1)),
              ("test", cut2, last)]
    split_rows = []
    for name, a, b in bounds:
        sel = (aqi_obs.index >= a) & (aqi_obs.index <= b)
        aqi_sel = daily["AQI"][sel]
        row = [name, str(a.date()), str(b.date()), int(sel.sum()), int(aqi_obs[sel].sum()),
               float(aqi_sel.median()), int((aqi_sel > ALERT_THRESHOLD).sum())]
        for L in (7, 14):
            w = windows[L]
            # target date AND whole history window must lie inside the split
            hist_start_ok = (w.index - pd.Timedelta(days=L)) >= a
            wsel = (w.index >= a) & (w.index <= b) & hist_start_ok
            row.append(int(w[wsel].sum()))
        split_rows.append(row)
    findings["split"] = pd.DataFrame(
        split_rows,
        columns=["split", "from", "to", "calendar_days", "AQI_observed_days",
                 "median_AQI", f"days_AQI_gt_{ALERT_THRESHOLD}", "samples_lookback_7", "samples_lookback_14"],
    )

    findings["rows_after_lockdown"] = int((kol["Date"] >= LOCKDOWN_START).sum())
    findings["lockdown_in_split"] = [n for n, a, b in bounds if a <= LOCKDOWN_START <= b]

    # Rows by year
    by_year = kol.groupby(kol["Date"].dt.year).agg(
        rows=("Date", "size"), aqi_observed=("AQI", "count"),
        median_aqi=("AQI", "median"))
    by_year.index.name = "year"
    findings["by_year"] = by_year

    # ---- figures ----------------------------------------------------------
    findings["figures"] = make_plots(daily, findings) if make_figures else []
    return findings


# ---------------------------------------------------------------------------
# Figures (optional - skipped politely if matplotlib is not installed)
# ---------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, RED_LINE = "#2a78d6", "#d03b3b"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def make_plots(daily: pd.DataFrame, f: dict) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("NOTE: matplotlib is not installed - figures skipped (report still written).")
        return []

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    })
    paths = []

    # 1. AQI time series - gaps are left as breaks (NOT interpolated)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(daily.index, daily["AQI"], color=BLUE, linewidth=1.4)
    ax.axhline(ALERT_THRESHOLD, color=RED_LINE, linewidth=1, linestyle="--")
    ax.text(daily.index[0], ALERT_THRESHOLD + 6, f"Alert threshold (AQI {ALERT_THRESHOLD})",
            color=INK2, fontsize=9)
    missing = daily.index[daily["AQI"].isna()]
    if len(missing):
        ymin = 0
        ax.scatter(missing, np.full(len(missing), ymin), marker="|", s=60,
                   color=INK2, linewidths=1, label="Day with no AQI value")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(bottom=-10)
    ax.set_ylabel("AQI")
    ax.set_title(f"Kolkata daily AQI, {f['first'].date()} to {f['last'].date()} "
                 "(gaps are real missing days, not interpolated)", loc="left", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "kolkata_aqi_timeseries.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 2. AQI category distribution
    bt = f["bucket_tbl"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bt["AQI_Bucket"], bt["days"], color=[BLUE] * 6 + ["#b5b4ae"], width=0.65)
    for b, d in zip(bars, bt["days"]):
        ax.text(b.get_x() + b.get_width() / 2, d + max(bt["days"]) * 0.01, str(d),
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Number of days")
    ax.set_title("Kolkata: days per AQI category (CPCB scale)", loc="left", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "kolkata_aqi_category_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 3. Missing % per column
    m = f["missing"].query("column not in ['City', 'Date']").sort_values("missing_pct_of_rows")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(m["column"], m["missing_pct_of_rows"], color=BLUE, height=0.6)
    for y, v in enumerate(m["missing_pct_of_rows"]):
        ax.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 105)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"% of the {f['n']} Kolkata rows with no value")
    ax.set_title("Kolkata: missing values per column", loc="left", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "kolkata_missing_values.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # 4. Coverage heatmap: column x month
    cov = f["col_month_cov"].T
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(cov.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_yticks(range(len(cov.index)), cov.index)
    step = max(1, len(cov.columns) // 14)
    ax.set_xticks(range(0, len(cov.columns), step), cov.columns[::step], rotation=45, ha="right")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("% of calendar days with a value")
    cb.outline.set_visible(False)
    ax.set_title("Kolkata: data coverage by column and month (white = nothing recorded)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "kolkata_coverage_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report(f: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    d = f["aqi_describe"]
    miss = f["missing"].copy()
    gap_runs = f["aqi_gap_runs"].sort_values("days", ascending=False)
    obs_runs = f["aqi_obs_runs"].sort_values("days", ascending=False)
    longest = obs_runs.iloc[0] if len(obs_runs) else None
    win = f["windows"]
    w7 = int(win.loc[win["lookback_days"] == 7, "samples_AQI_only"].iloc[0])
    w1 = int(win.loc[win["lookback_days"] == 1, "samples_AQI_only"].iloc[0])

    L = []
    a = L.append
    a("# Kolkata Dataset Audit")
    a("")
    a(f"*Generated by `src/inspect_data.py` on {datetime.now():%Y-%m-%d %H:%M}. "
      "Every number below is computed from the CSV at run time. Re-running the script "
      "regenerates this file.*")
    a("")
    a("## 0. Source file")
    a("")
    a(f"- File: `{rel(f['input'])}` (read-only; not modified by this script)")
    a(f"- SHA-256: `{f['sha256']}` (fingerprint - if this changes, the raw file has changed)")
    a("- Source: Kaggle, *Air Quality Data in India* by Rohan Rao "
      "(https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)")
    a(f"- Whole file: {f['all_rows']:,} rows, {f['all_cities']} cities, "
      f"{f['all_date_range'][0]} to {f['all_date_range'][1]}")
    a(f"- Columns: {', '.join(f'`{c}`' for c in f['columns'])}")
    a("- Units are not stated inside the CSV itself.")
    a("")

    a("## 1-2. Kolkata observations and date coverage")
    a("")
    a(f"- **Kolkata rows:** {f['n']}")
    a(f"- **First date:** {f['first'].date()}   **Last date:** {f['last'].date()}")
    a(f"- **Calendar days in that range:** {f['calendar_days']}")
    a(f"- **Days with an AQI value:** {f['aqi_observed_days']} "
      f"({100 * f['aqi_observed_days'] / f['calendar_days']:.1f}% of calendar days)")
    a("")
    a("Rows per year:")
    a("")
    a(md_table(f["by_year"], index=True, floatfmt="{:.1f}"))
    a("")
    a("> Note: other cities in the same file start in 2015. Kolkata only starts in "
      f"{f['first'].year}, so Kolkata has roughly {f['calendar_days'] / 365.25:.1f} years of history.")
    a("")

    a("## 3-4. Missing values per column")
    a("")
    a(f"Percentages are out of the {f['n']} Kolkata rows that exist. "
      "(Dates with no row at all are counted separately in section 8.)")
    a("")
    a(md_table(miss[["column", "missing_count", "missing_pct_of_rows", "non_missing_count"]],
               floatfmt="{:.1f}"))
    a("")
    if f["never_measured"]:
        a(f"- **Never measured for Kolkata (100% missing):** {', '.join(f['never_measured'])}")
    heavy = [h for h in f["heavy_missing"] if h not in f["never_measured"]]
    if heavy:
        a(f"- **Excessively incomplete (> {HEAVY_MISSING_PCT:.0f}% missing):** {', '.join(heavy)}")
    a(f"- **Pollutants with <= {HEAVY_MISSING_PCT:.0f}% missing:** "
      f"{', '.join(f['usable_pollutants']) or 'none'}")
    a("")

    a("## 5. Pollutant columns")
    a("")
    a(f"The CSV has {len(f['pollutants'])} pollutant columns: "
      f"{', '.join(f['pollutants'])}. There are **no weather columns** "
      "(no temperature, humidity, wind or rainfall).")
    a("")

    a("## 6. AQI distribution")
    a("")
    a("| statistic | value |\n|---|---|")
    for k in ["count", "mean", "std", "min", "5%", "25%", "50%", "75%", "95%", "max"]:
        a(f"| {k} | {d[k]:.1f} |")
    a("")
    a(md_table(f["bucket_tbl"], floatfmt="{:.1f}"))
    a("")
    a(f"- Days with AQI above the project alert threshold ({ALERT_THRESHOLD}): "
      f"**{f['days_above_alert']}** "
      f"({100 * f['days_above_alert'] / max(f['aqi_observed_days'], 1):.1f}% of days that have an AQI).")
    a("")
    a("Typical AQI by calendar month (all years pooled - descriptive only, not a cause):")
    a("")
    a(md_table(f["by_moy"], index=True, floatfmt="{:.1f}"))
    a("")

    a("## 7. Duplicates")
    a("")
    a(f"- Fully identical duplicate rows: **{f['dup_full_rows']}**")
    a(f"- Dates appearing more than once: **{f['dup_dates']}**")
    a("")

    a("## 8. Missing dates and gaps")
    a("")
    a(f"- Calendar dates with **no row at all**: **{f['absent_dates']}**")
    if f["absent_dates"]:
        a("")
        a(md_table(f["absent_runs"].assign(start=lambda x: x["start"].dt.date,
                                           end=lambda x: x["end"].dt.date)))
    a(f"- Calendar days where **AQI is missing** (row absent or AQI blank): "
      f"**{f['aqi_missing_days_calendar']}**, in {len(gap_runs)} separate gap(s).")
    a("")
    if len(gap_runs):
        a("Longest AQI gaps:")
        a("")
        a(md_table(gap_runs.head(10).assign(start=lambda x: x["start"].dt.date,
                                            end=lambda x: x["end"].dt.date)))
        a("")
    if longest is not None:
        a(f"- Longest unbroken stretch of AQI values: **{int(longest['days'])} days** "
          f"({longest['start'].date()} to {longest['end'].date()}).")
        a(f"- Number of unbroken stretches: {len(obs_runs)}")
    a("")
    a("Monthly coverage (AQI):")
    a("")
    a(md_table(f["monthly"], index=True, floatfmt="{:.1f}"))
    a("")

    a("## Data quality and plausibility checks")
    a("")
    a(md_table(f["anom"], floatfmt="{:.2f}"))
    a("")
    a("How to read this table: *negative* values are physically impossible for concentrations; "
      "*exact_zero* may mean 'below detection limit' or a recording fault; *far outliers* are "
      "values unusually high compared with the rest of the column (they may still be real "
      "pollution episodes - they are flagged, not removed); a long *identical run* (same value "
      "several days in a row) can indicate a stuck sensor or copied values.")
    a("")
    a(f"- AQI outside the 0-500 scale: **{f['aqi_out_of_range']}**")
    a(f"- Days where PM2.5 > PM10 (suspicious, since PM2.5 is a subset of PM10): "
      f"**{f['pm25_gt_pm10']}** of {f['pm_both_present']} days with both values"
      + (f" - dates: {', '.join(map(str, f['pm25_gt_pm10_dates'][:15]))}"
         + (" ..." if len(f['pm25_gt_pm10_dates']) > 15 else "") if f['pm25_gt_pm10'] else ""))
    a(f"- `AQI_Bucket` disagrees with the CPCB category of the `AQI` number: **{f['bucket_mismatch']}** rows")
    a(f"- AQI present but bucket missing: {f['aqi_without_bucket']}; "
      f"bucket present but AQI missing: {f['bucket_without_aqi']}")
    c = f["aqi_daily_change"]
    a(f"- Day-to-day AQI change (absolute, consecutive observed days only): median "
      f"{c['50%']:.1f}, 90th percentile {c['90%']:.1f}, max {c['max']:.1f}")
    a("")
    a("Largest day-to-day AQI changes (flagged for checking, not removed):")
    a("")
    a(md_table(f["top_jumps"], floatfmt="{:.1f}"))
    a("")
    a("Highest-AQI days:")
    a("")
    a(md_table(f["top_aqi_days"], floatfmt="{:.1f}"))
    a("")
    a("Lowest-AQI days:")
    a("")
    a(md_table(f["bottom_aqi_days"], floatfmt="{:.1f}"))
    a("")

    a("## 9. Enough consecutive history for next-day forecasting?")
    a("")
    a("A *sample* = a block of past days (the lookback) followed by the next day's AQI "
      "(the target). Here a sample is counted **only if every day in the block and the "
      "target day have a real recorded value** - no gap filling. These are therefore the "
      "minimum numbers available; imputation (documented separately) could raise them.")
    a("")
    a(md_table(win))
    a("")
    a(f"*Usable pollutants* here = {', '.join(f['usable_pollutants'])} "
      f"(each <= {HEAVY_MISSING_PCT:.0f}% missing). Days where AQI and all of these are present: "
      f"{f['multi_observed_days']}.")
    a("")

    a("## 10. Enough for chronological train / validation / test?")
    a("")
    a(f"Illustrative split by calendar time ({int(SPLIT_FRACTIONS[0]*100)}% / "
      f"{int(SPLIT_FRACTIONS[1]*100)}% / {int(SPLIT_FRACTIONS[2]*100)}%), oldest data for "
      "training and newest for testing. A sample is only counted in a split if its whole "
      "history window and its target lie inside that split (so no information leaks "
      "across the boundary). This split is for the audit only - the final split is a "
      "preprocessing decision still to be agreed.")
    a("")
    a(md_table(f["split"]))
    a("")
    a(f"- Rows on or after 2020-03-25 (start of India's national COVID-19 lockdown - "
      f"external context, not in the CSV): **{f['rows_after_lockdown']}**, falling in the "
      f"**{', '.join(f['lockdown_in_split']) or 'none'}** split of this illustration.")
    a("")

    a("## Figures")
    a("")
    if f["figures"]:
        for p in f["figures"]:
            a(f"- `{rel(p)}`")
    else:
        a("- Not generated (matplotlib not installed or `--no-figures` used).")
    a("")

    a("## What this audit did not do")
    a("")
    a("- No values were filled in, removed, or changed. `data/raw/city_day.csv` is untouched.")
    a("- No weather or other external data was added.")
    a("- No model was trained, so nothing here says how accurate a forecast will be.")
    a("- Monthly / seasonal AQI patterns above are descriptive correlations with the calendar, "
      "not evidence of what causes pollution.")
    a("")

    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
def print_summary(f: dict) -> None:
    print(f"Raw file        : {f['input']}")
    print(f"SHA-256         : {f['sha256']}")
    print(f"Kolkata rows    : {f['n']}")
    print(f"Date range      : {f['first'].date()} -> {f['last'].date()} "
          f"({f['calendar_days']} calendar days)")
    print(f"AQI observed    : {f['aqi_observed_days']} days; "
          f"AQI missing on {f['aqi_missing_days_calendar']} calendar days")
    print(f"Absent dates    : {f['absent_dates']}   Duplicate dates: {f['dup_dates']}")
    print(f"Never measured  : {', '.join(f['never_measured']) or 'none'}")
    print(f"Next-day samples (AQI only, no imputation): "
          + ", ".join(f"L={r.lookback_days}: {r.samples_AQI_only}" for r in f["windows"].itertuples()))
    print(f"Report written  : {REPORT_PATH}")
    for p in f["figures"]:
        print(f"Figure written  : {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Kolkata subset of city_day.csv")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="path to city_day.csv (default: data/raw/city_day.csv)")
    parser.add_argument("--no-figures", action="store_true", help="skip creating figures")
    args = parser.parse_args()

    findings = run_audit(args.input, make_figures=not args.no_figures)
    write_report(findings)
    print_summary(findings)


if __name__ == "__main__":
    main()
