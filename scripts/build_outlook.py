"""Builds the website's future outlook: rolls the live GRU forward day by day and measures its error by backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# File locations, outlook end date, backtest settings and error buckets.
ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "kolkata_extended_daily.csv"
META = ROOT / "models" / "live" / "gru_live.json"
WEIGHTS = ROOT / "models" / "live" / "gru_live.npz"
OUT = ROOT / "web" / "data" / "outlook.json"
BULLETIN = ROOT / "data" / "live" / "bulletin_aqi.csv"
WEATHER_RECENT = ROOT / "data" / "live" / "weather_recent.csv"
OUTLOOK_END = "2027-12-31"
BACKTEST_START, BACKTEST_STEP, BACKTEST_MAX = "2025-01-01", 7, 365
BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 120), (121, 365), (366, 430)]
SHOW_WEATHER = ["temp_mean", "temp_min", "temp_max", "humidity", "rain_mm", "wind_mean"]


def load_model():
    """Load the live GRU settings and the weights of every seed."""
    meta = json.loads(META.read_text())
    raw = np.load(WEIGHTS)
    keys = ("weight_ih", "weight_hh", "bias_ih", "bias_hh", "head_w", "head_b")
    return meta, [{k: raw[f"s{i}_{k}"] for k in keys} for i in range(meta["seeds"])]


def gru_batch(w, x):
    """Run the GRU and its output layer in NumPy for a batch of 14-day windows."""
    H = w["weight_hh"].shape[1]
    h = np.zeros((x.shape[0], H))
    for step in range(x.shape[1]):
        gi = x[:, step] @ w["weight_ih"].T + w["bias_ih"]
        gh = h @ w["weight_hh"].T + w["bias_hh"]
        r = 1 / (1 + np.exp(-(gi[:, :H] + gh[:, :H])))
        z = 1 / (1 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H])))
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1 - z) * n + z * h
    return h @ w["head_w"][0] + w["head_b"][0]


def climatology(t: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Average each column per calendar day over all years, smoothed over 15 days."""
    doy = t["Date"].dt.day_of_year.clip(upper=365)
    c = t.groupby(doy)[cols].mean().reindex(range(1, 366)).interpolate()
    c = pd.concat([c.iloc[-7:], c, c.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    c.index = range(1, 366)
    return c


def day_features(dates: pd.DatetimeIndex, clim: pd.DataFrame, weather_cols: list[str],
                 known: pd.DataFrame | None = None) -> np.ndarray:
    """Build each date's weather and day-of-year inputs, using real weather where known."""
    doy = np.minimum(dates.day_of_year, 365)
    ang = 2 * np.pi * (dates.day_of_year - 1) / np.where(dates.is_leap_year, 366, 365)
    w = clim.loc[doy, weather_cols].to_numpy(float)
    if known is not None:
        k = known.reindex(dates)[weather_cols].to_numpy(float)
        w = np.where(np.isnan(k), w, k)
    return np.column_stack([w, np.sin(ang), np.cos(ang)])


def roll(meta, seeds, clim, hist_aqi: np.ndarray, first_dates: list[pd.Timestamp], n: int,
         known: pd.DataFrame | None = None) -> np.ndarray:
    """Forecast day after day from several start points, feeding each forecast back in."""
    L = meta["lookback"]
    weather_cols = meta["features"][1:-2]
    mean, std, s = np.array(meta["feature_mean"]), np.array(meta["feature_std"]), meta["target"]
    B = len(first_dates)
    feats = np.stack([day_features(pd.date_range(d - pd.Timedelta(days=L), periods=L + n), clim, weather_cols, known)
                      for d in first_dates])
    aqi = np.zeros((B, L + n))
    aqi[:, :L] = hist_aqi
    for k in range(n):
        x = np.concatenate([aqi[:, k:k + L, None], feats[:, k:k + L]], axis=2)
        x = (x - mean) / std
        last = aqi[:, k + L - 1]
        preds = [gru_batch(w, x) for w in seeds]
        if meta["mode"] == "level":
            aqi[:, k + L] = np.mean([p * s["std"] + s["mean"] for p in preds], axis=0)
        else:
            aqi[:, k + L] = np.mean([last + p * s["dstd"] for p in preds], axis=0)
    return aqi[:, L:]


def main() -> None:
    """Backtest the rolling forecast, check it on recent bulletins, and write outlook.json."""
    meta, seeds = load_model()
    t = pd.read_csv(TABLE, parse_dates=["Date"])
    weather_cols = meta["features"][1:-2]
    L = meta["lookback"]
    aqi = t.set_index("Date")["AQI"]
    target = t.set_index("Date")["AQI_target"]

    past = t[t["Date"] <= "2024-12-31"]
    clim_bt = climatology(past, weather_cols)
    season_bt = climatology(past.assign(a=past["AQI"]), ["a"])["a"]
    last_day = t["Date"].max()
    starts = pd.date_range(BACKTEST_START, last_day - pd.Timedelta(days=1), freq=f"{BACKTEST_STEP}D")
    hist = np.stack([aqi.loc[d - pd.Timedelta(days=L): d - pd.Timedelta(days=1)].to_numpy() for d in starts])
    fc = roll(meta, seeds, clim_bt, hist, list(starts), BACKTEST_MAX)
    errs = {h: [] for h in range(1, BACKTEST_MAX + 1)}
    base = {h: [] for h in range(1, BACKTEST_MAX + 1)}
    for i, d in enumerate(starts):
        for h in range(1, BACKTEST_MAX + 1):
            day = d + pd.Timedelta(days=h - 1)
            if day > last_day or np.isnan(target.get(day, np.nan)):
                continue
            errs[h].append((target[day], fc[i, h - 1]))
            base[h].append(abs(target[day] - season_bt.loc[min(day.day_of_year, 365)]))
    buckets = []
    for lo, hi in BUCKETS:
        pairs = [p for h in range(lo, min(hi, BACKTEST_MAX) + 1) for p in errs[h]]
        e = np.array([a - f for a, f in pairs])
        ratio = np.array([a / f for a, f in pairs if f > 0])
        b = np.concatenate([base[h] for h in range(lo, min(hi, BACKTEST_MAX) + 1)] or [[]])
        if len(e) < 30:
            buckets.append({**buckets[-1], "from": lo, "to": hi, "extrapolated": True})
            continue
        buckets.append({"from": lo, "to": hi, "n": int(len(e)), "mae": round(float(np.mean(np.abs(e))), 1),
                        "r10": round(float(np.percentile(ratio, 10)), 3), "r90": round(float(np.percentile(ratio, 90)), 3),
                        "seasonal_avg_mae": round(float(np.mean(b)), 1)})
        print(f"  {lo:>3}-{hi:<3} days ahead: MAE {buckets[-1]['mae']:5.1f}  "
              f"(seasonal average {buckets[-1]['seasonal_avg_mae']:5.1f})  n={len(e)}")

    clim = climatology(t, weather_cols)
    clim_show = climatology(t, SHOW_WEATHER)
    known = None
    if WEATHER_RECENT.exists():
        import sys
        sys.path.insert(0, str(ROOT))
        from src.weather import load_weather
        known = load_weather(WEATHER_RECENT).set_index("Date")
        known = known[known.index > t["Date"].max()]
        print(f"Real/forecast weather: {known.index.min().date()} to {known.index.max().date()}")

    recent, check = [], {}
    if BULLETIN.exists():
        b = pd.read_csv(BULLETIN, parse_dates=["date"]).set_index("date")["aqi"].astype(float)
        b = b[b.index > last_day]
        if len(b) and (b.index == pd.date_range(last_day + pd.Timedelta(days=1), periods=len(b))).all():
            full = pd.concat([aqi.loc[:last_day], b])
            rd = list(b.index)
            hist_r = np.stack([full.loc[d - pd.Timedelta(days=L): d - pd.Timedelta(days=1)].to_numpy() for d in rd])
            day_ahead = roll(meta, seeds, clim, hist_r, rd, 1, known)[:, 0]
            prev = roll(meta, seeds, clim, aqi.loc[:last_day].tail(L).to_numpy()[None],
                        [last_day + pd.Timedelta(days=1)], len(rd))[0]
            pers = full.shift(1).loc[rd].to_numpy()
            for d, y, f1, fp, p0 in zip(rd, b.to_numpy(), day_ahead, prev, pers):
                wx = clim_show.loc[min(d.day_of_year, 365)]
                recent.append({"date": d.strftime("%Y-%m-%d"), "real": True, "aqi": float(y),
                               "day_ahead": round(float(f1), 1), "prev": round(float(fp), 1),
                               "persistence": float(p0), **{k: round(float(wx[k]), 1) for k in SHOW_WEATHER}})
            y = b.to_numpy()
            check = {"from": rd[0].strftime("%Y-%m-%d"), "to": rd[-1].strftime("%Y-%m-%d"), "days": len(rd),
                     "day_ahead_mae": round(float(np.mean(np.abs(day_ahead - y))), 1),
                     "persistence_mae": round(float(np.mean(np.abs(pers - y))), 1),
                     "old_outlook_mae": round(float(np.mean(np.abs(prev - y))), 1)}
            print(f"Bulletins {check['from']} to {check['to']}: day-ahead MAE {check['day_ahead_mae']}, "
                  f"persistence {check['persistence_mae']}, outlook from {last_day.date()} {check['old_outlook_mae']}")
            aqi, last_day = full, rd[-1]
        else:
            print("Bulletin file does not continue the main data day by day - ignored.")

    first = last_day + pd.Timedelta(days=1)
    dates = pd.date_range(first, OUTLOOK_END)
    fc = roll(meta, seeds, clim, aqi.loc[:last_day].tail(L).to_numpy()[None], [first], len(dates), known)[0]
    md = t.assign(md=t["Date"].dt.strftime("%m-%d"), y=t["Date"].dt.year).pivot_table(
        index="md", columns="y", values="AQI_target")
    days = []
    for h, (d, p) in enumerate(zip(dates, fc), start=1):
        b = next(x for x in buckets if x["from"] <= h <= x["to"])
        past_years = md.loc[d.strftime("%m-%d")].dropna() if d.strftime("%m-%d") in md.index else pd.Series()
        wx = clim_show.loc[min(d.day_of_year, 365)]
        days.append({"date": d.strftime("%Y-%m-%d"), "h": h, "aqi": round(float(p), 1),
                     "low": round(float(p * b["r10"]), 1), "high": round(float(p * b["r90"]), 1),
                     "mae": b["mae"],
                     "past": {str(int(y)): round(float(v)) for y, v in past_years.items()},
                     **{k: round(float(wx[k]), 1) for k in SHOW_WEATHER}})
    for r in recent:
        r["past"] = {str(int(y)): round(float(v)) for y, v in md.loc[r["date"][5:]].dropna().items()}
    OUT.write_text(json.dumps({
        "data_from": t["Date"].min().strftime("%Y-%m-%d"), "data_to": last_day.strftime("%Y-%m-%d"),
        "main_data_to": t["Date"].max().strftime("%Y-%m-%d"), "check": check, "recent": recent,
        "model": "Live GRU (city AQI + weather + day of year), rolled forward day by day",
        "backtest": f"Started every {BACKTEST_STEP} days from {BACKTEST_START}, climatology from 2018-2024 only",
        "buckets": buckets, "days": days}, separators=(",", ":")))
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(days)} days, {days[0]['date']} to {days[-1]['date']}")


if __name__ == "__main__":
    main()
