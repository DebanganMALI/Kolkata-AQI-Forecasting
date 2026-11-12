"""
Extended dataset: Kolkata, 2018-10-01 .. 2026-10-31, from official sources.

  target     official CPCB city AQI (daily 4 PM bulletin), data/raw/cpcb/city_aqi/
             -> converted once to data/interim/cpcb_kolkata_city_aqi_daily.csv
  pollutants 7 WBPCB stations (CPCB CAAQMS repository, 24-hour files),
             averaged over the stations reporting each day
  weather    Open-Meteo, 12 daily variables (same as the main comparison)
  fires      NASA FIRMS VIIRS S-NPP vegetation fires, daily count and power
  calendar   day of year, festivals (Kali Puja, Durga Puja, Holi, New Year) and
             the 2020 national lockdown

Rules (as in the rest of the project):
  * Raw files are only read. The built table goes to data/processed/.
  * The TARGET is never filled in: a day without an official AQI is never used
    as a forecast target. As an INPUT, a missing value is carried forward from
    the previous day (at most config.V2_FFILL_LIMIT days) and flagged.
  * Weather is never filled in. A day without fire detections has 0 fires.
  * Festival and lockdown flags describe the FORECAST day ("tomorrow"): the
    calendar is known in advance, so this is not future information.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.weather import load_weather

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
STATION_COLUMNS = {p: p for p in POLLUTANTS if p != "O3"} | {"Ozone": "O3"}


def convert_city_aqi_xlsx(src: Path = config.V2_CITY_AQI_XLSX_DIR,
                          out: Path = config.V2_CITY_AQI_CSV) -> pd.DataFrame:
    """CPCB city AQI workbooks (days down, months across) -> one Date, AQI table."""
    rows = []
    for f in sorted(Path(src).glob("AQI_Data_City_Level_Kolkata_*_Daily.xlsx")):
        year = int(re.search(r"_(\d{4})_Daily", f.name).group(1))
        sheet = pd.read_excel(f, engine="openpyxl")
        for m in range(1, 13):
            for _, r in sheet.iterrows():
                d = pd.to_numeric(r.iloc[0], errors="coerce")
                if pd.notna(d) and int(d) <= calendar.monthrange(year, m)[1]:
                    rows.append({"Date": f"{year}-{m:02d}-{int(d):02d}",
                                 "AQI": pd.to_numeric(r.iloc[m], errors="coerce")})
    df = pd.DataFrame(rows).drop_duplicates("Date").sort_values("Date")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_city_aqi() -> pd.Series:
    if not config.V2_CITY_AQI_CSV.exists():
        convert_city_aqi_xlsx()           # needs openpyxl
    df = pd.read_csv(config.V2_CITY_AQI_CSV, parse_dates=["Date"])
    return df.set_index("Date")["AQI"]


def load_stations() -> tuple[pd.DataFrame, pd.Series]:
    """Daily mean of each pollutant over the stations reporting it, + PM2.5 station count."""
    frames = []
    for f in sorted(config.V2_STATIONS_DIR.glob("Raw_Data_*_1D.csv")):
        station = re.search(r"site_\d+_(.+?)_kolkata", f.name).group(1)
        df = pd.read_csv(f)
        df.columns = [c.split(" (")[0].strip() for c in df.columns]
        keep = {k: v for k, v in STATION_COLUMNS.items() if k in df.columns}
        out = df[["Timestamp", *keep]].rename(columns={"Timestamp": "Date", **keep})
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        for c in keep.values():
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["station"] = station
        frames.append(out)
    long = pd.concat(frames, ignore_index=True)
    city = long.groupby("Date")[POLLUTANTS].mean()
    n_pm = long.dropna(subset=["PM2.5"]).groupby("Date")["station"].nunique()
    return city, n_pm


def load_fires(days: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.read_csv(config.V2_FIRES_CSV, parse_dates=["Date"]).set_index("Date").reindex(days)
    f = f.fillna(0.0)                      # no detection that day = 0 fires
    return pd.DataFrame({"log_fires_300km": np.log1p(f["fires_300km"]),
                         "log_frp_region": np.log1p(f["frp_region_MW"])}, index=days)


def event_flags(days: pd.DatetimeIndex) -> pd.DataFrame:
    """1 if the NEXT day (the forecast day) falls in the event window."""
    nxt = days + pd.Timedelta(days=1)
    out = pd.DataFrame(index=days)
    for name, (before, after) in config.V2_EVENT_WINDOWS.items():
        flag = np.zeros(len(days), dtype=float)
        for d in config.V2_EVENT_DATES[name]:
            d = pd.Timestamp(d)
            flag[(nxt >= d - pd.Timedelta(days=before)) & (nxt <= d + pd.Timedelta(days=after))] = 1.0
        out[f"{name}_next"] = flag
    lo, hi = (pd.Timestamp(x) for x in config.V2_LOCKDOWN)
    out["lockdown_next"] = ((nxt >= lo) & (nxt <= hi)).astype(float)
    return out


def assign_split_v2(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    t, v = (np.datetime64(pd.Timestamp(x)) for x in (config.V2_TRAIN_END, config.V2_VAL_END))
    return np.where(d <= t, "train", np.where(d <= v, "val", "test"))


def build_extended_table(save: bool = True) -> tuple[pd.DataFrame, dict]:
    days = pd.date_range(config.V2_START, config.V2_END, freq="D")
    log: dict = {"days": len(days)}
    t = pd.DataFrame(index=days)
    t.index.name = "Date"

    aqi = load_city_aqi().reindex(days)
    t["AQI_target"] = aqi                                  # never filled
    t["AQI"] = aqi.ffill(limit=config.V2_FFILL_LIMIT)      # input copy
    t["AQI_imputed"] = aqi.isna()
    log["target_missing_days"] = int(aqi.isna().sum())

    city, n_pm = load_stations()
    city = city.reindex(days)
    log["filled_input_cells"] = {}
    for c in POLLUTANTS:
        t[f"{c}_imputed"] = city[c].isna()
        t[c] = city[c].ffill(limit=config.V2_FFILL_LIMIT)
        log["filled_input_cells"][c] = int(city[c].isna().sum())
    t["n_stations_pm25"] = n_pm.reindex(days).fillna(0).astype(int)

    w = load_weather(config.V2_WEATHER_CSV).set_index("Date").reindex(days)
    t[list(config.WEATHER_FEATURES)] = w[list(config.WEATHER_FEATURES)]
    t = t.join(load_fires(days)).join(event_flags(days))

    doy = days.day_of_year.to_numpy()
    angle = 2 * np.pi * (doy - 1) / np.where(days.is_leap_year, 366, 365)
    t["doy_sin"], t["doy_cos"] = np.sin(angle), np.cos(angle)
    t["split"] = assign_split_v2(days)

    inputs = ["AQI", *POLLUTANTS, *config.WEATHER_FEATURES, *config.V2_FIRE_FEATURES]
    bad = t[inputs].isna().any(axis=1)
    if bad.any():
        cols = t.loc[bad, inputs].isna().sum()
        raise ValueError(f"{int(bad.sum())} day(s) still have missing inputs after carrying forward "
                         f"at most {config.V2_FFILL_LIMIT} days: {cols[cols > 0].to_dict()} "
                         f"(first {t.index[bad][0].date()}). Nothing is invented beyond that limit.")
    t = t.reset_index()
    log["split_days"] = t["split"].value_counts().reindex(["train", "val", "test"]).to_dict()
    if save:
        config.V2_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
        t.assign(Date=t["Date"].dt.strftime("%Y-%m-%d")).to_csv(config.V2_TABLE_CSV, index=False)
    return t, log
