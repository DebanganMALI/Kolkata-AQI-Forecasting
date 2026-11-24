"""Live presentation demo: trains the GRU on the GPU, scores the test days and forecasts tomorrow."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make the project and scripts folders importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import compare, config, extended
from src.evaluate import block_bootstrap_mae_diff
from src.model import AQIGRU, pick_device

# Time zone, run mode, chosen model and AQI categories.
IST = timezone(timedelta(hours=5, minutes=30))
FAST = "--fast" in sys.argv
FEATURE_SET, FAMILY = "pollution+weather", "GRU"
CANDIDATE = next(c for c in config.V2_RNN_CANDIDATES if c["name"] == "level_small")
CATS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (10_000, "Severe")]


def cat(v: float) -> str:
    """Return the CPCB category name for an AQI value."""
    return next(name for top, name in CATS if v <= top)


def step(n: int, title: str) -> None:
    """Print a step header, pausing first unless --fast is used."""
    if not FAST and n > 1:
        input("\n  Press Enter for the next step... ")
    print("\n" + "=" * 76 + f"\n  STEP {n}.  {title}\n" + "=" * 76)


def row(label: str, value: str) -> None:
    """Print one aligned label and value line."""
    print(f"  {label:<46}{value}")


def load_data():
    """Load the stored 2018-2026 table, build 14-day windows and show their sizes."""
    table = pd.read_csv(config.V2_TABLE_CSV, parse_dates=["Date"])
    data = extended.build_data(table, FEATURE_SET)
    w = data[0]
    print(f"  File: {config.V2_TABLE_CSV.relative_to(ROOT)}")
    row("Days of data", f"{len(table):,}  ({table['Date'].min():%d %b %Y} to {table['Date'].max():%d %b %Y})")
    row("Inputs per day", f"{w.X.shape[2]}  (AQI, 11 pollutants, 13 weather, day of year)")
    row("Window", f"{w.X.shape[1]} days of history -> AQI tomorrow")
    for s, name in (("train", "Training samples (2018-2023)"), ("val", "Validation samples (2024)"),
                    ("test", "Test samples (2025-2026, never seen)")):
        row(name, f"{(w.split == s).sum():,}")
    return data


def train_and_test(data, device):
    """Train the chosen GRU for each seed on the GPU and average its test forecasts."""
    print(f"  Model: GRU, 1 layer x {CANDIDATE['hidden_size']} units, predicts the AQI level, "
          f"device: {str(device).upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    preds, models, t0 = [], [], time.time()
    for seed in config.V2_SEEDS:
        t = time.time()
        out, model = compare.fit_predict(FAMILY, CANDIDATE, seed, data, ["test"], device)
        preds.append(out["test"])
        models.append(model)
        print(f"    seed {seed}: trained in {time.time() - t:5.1f} s")
    print(f"  Training finished in {time.time() - t0:.0f} s")
    return np.mean(preds, axis=0)


def score(data, pred):
    """Score the test forecasts against persistence and show example days."""
    w = data[0].subset("test")
    y, pers = w.y_true, w.last_aqi
    mae, pmae = np.mean(np.abs(pred - y)), np.mean(np.abs(pers - y))
    r2 = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    diff, lo, hi = block_bootstrap_mae_diff(y, pred, pers)
    row("Persistence (tomorrow = today), error", f"{pmae:.1f} AQI points")
    row("GRU, just trained on this GPU, error", f"{mae:.1f} AQI points")
    row("Improvement [95% interval]", f"{diff:+.1f}  [{lo:+.1f}, {hi:+.1f}]"
        + ("  -> better than persistence" if hi < 0 else ""))
    row("R squared", f"{r2:.3f}")
    row("Forecasts within 20 points", f"{np.mean(np.abs(pred - y) <= 20):.0%} of days")
    dates = pd.to_datetime(w.target_dates)
    pick = [i for i, d in enumerate(dates) if d.strftime("%Y-%m-%d") in
            ("2025-01-14", "2025-07-20", "2025-11-18", "2025-12-12", "2026-01-08", "2026-05-01", "2026-10-31")]
    print("\n    Date          Actual    GRU   Error   Persistence")
    for i in pick:
        print(f"    {dates[i]:%d %b %Y}    {y[i]:5.0f}  {pred[i]:5.0f}  {pred[i] - y[i]:+5.0f}   {pers[i]:5.0f}")


def tomorrow(device):
    """Fetch the newest data and forecast tomorrow with the live GRU on the GPU."""
    import build_outlook as B
    import update_live
    print("  Downloading the newest CPCB bulletins and weather...")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            update_live.download_bulletins()
            update_live.download_weather()
        except Exception as e:
            print(f"  (download failed: {e}; using the saved data)")
        update_live.read_bulletins.main()
    meta, seeds = B.load_model()
    t = pd.read_csv(B.TABLE, parse_dates=["Date"])
    weather_cols = meta["features"][1:-2]
    b = pd.read_csv(B.BULLETIN, parse_dates=["date"]).set_index("date")["aqi"].astype(float)
    aqi = pd.concat([t.set_index("Date")["AQI"], b[b.index > t["Date"].max()]])
    last = aqi.index.max()
    known = None
    if B.WEATHER_RECENT.exists():
        from src.weather import load_weather
        known = load_weather(B.WEATHER_RECENT).set_index("Date")
    clim = B.climatology(t, weather_cols)
    L = meta["lookback"]
    days = pd.date_range(last - pd.Timedelta(days=L - 1), last)
    feats = B.day_features(days, clim, weather_cols, known)
    x = np.column_stack([aqi.reindex(days).ffill().to_numpy(), feats])
    x = (x - np.array(meta["feature_mean"])) / np.array(meta["feature_std"])
    xt = torch.tensor(x[None], dtype=torch.float32, device=device)
    s, out = meta["target"], []
    for w in seeds:
        m = AQIGRU(len(meta["features"]), hidden_size=w["weight_hh"].shape[1], num_layers=1)
        m.load_state_dict({"gru.weight_ih_l0": torch.tensor(w["weight_ih"]), "gru.weight_hh_l0": torch.tensor(w["weight_hh"]),
                           "gru.bias_ih_l0": torch.tensor(w["bias_ih"]), "gru.bias_hh_l0": torch.tensor(w["bias_hh"]),
                           "head.1.weight": torch.tensor(w["head_w"]), "head.1.bias": torch.tensor(w["head_b"])})
        m = m.float().to(device).eval()
        with torch.no_grad():
            z = m(xt).item()
        out.append(z * s["std"] + s["mean"] if meta["mode"] == "level" else aqi[last] + z * s["dstd"])
    p = float(np.mean(out))
    nxt = last + pd.Timedelta(days=1)
    print(f"\n  Newest official CPCB value: {aqi[last]:.0f} ({cat(aqi[last])}) on {last:%d %b %Y}")
    print("  Last 7 official days: " + ", ".join(f"{d:%d %b} {v:.0f}" for d, v in aqi.tail(7).items()))
    print(f"\n  GRU FORECAST FOR {nxt:%A %d %B %Y}:  {p:.0f}  ({cat(p)})   typical error +/-13")
    return nxt, p


def government_now(key):
    """Fetch the live CPCB feed and print the government AQI right now."""
    import urllib.parse
    from live_forecast import city_aqi_from_records
    from update_live import FEED, fetch
    q = urllib.parse.urlencode({"api-key": key, "format": "json", "limit": 500, "filters[city]": "Kolkata"})
    try:
        records = json.loads(fetch(f"{FEED}?{q}"))["records"]
    except Exception as e:
        print(f"  Could not reach data.gov.in: {e}")
        return None
    aqi, n = city_aqi_from_records(records)
    stamp = max(r.get("last_update", "") for r in records)
    if aqi is not None:
        print(f"  GOVERNMENT AQI RIGHT NOW: {aqi:.0f} ({cat(aqi)})  from {n} CPCB stations, feed time {stamp}")
        print("  Indian AQI, rolling 24-hour average (apps showing US AQI give different numbers).")
    return aqi


def main() -> None:
    """Run all demo steps in order."""
    print("\n  KOLKATA NEXT-DAY AQI FORECASTING  -  live run")
    print(f"  {datetime.now(IST):%A %d %B %Y, %I:%M %p} IST")
    key = os.environ.get("DATA_GOV_API_KEY") or input("\n  data.gov.in API key: ").strip()
    device = pick_device(config.DEVICE)

    step(1, "Load the stored data");                                    data = load_data()
    step(2, "Train the GRU from scratch on the GPU");                   pred = train_and_test(data, device)
    step(3, "Predict the 669 test days it never saw");                  score(data, pred)
    step(4, "Forecast tomorrow from the newest official data");         nxt, p = tomorrow(device)
    step(5, "Compare with the government AQI right now");               now = government_now(key)
    if now is not None:
        print(f"\n  Government now {now:.0f}  vs  model forecast for {nxt:%d %b} {p:.0f}  "
              f"(difference {p - now:+.0f}; the official 4 PM value is the fair comparison)")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
