"""
Use the trained LSTM to forecast AQI for one day and raise an alert if needed.

Two modes:
  * no date given  -> forecast the day AFTER the last date in the dataset
                      (2020-07-02). There is no actual value to compare with.
  * --date YYYY-MM-DD -> forecast that date from the 14 days before it, and show
                      the real AQI next to it if the dataset has it.

Note: the dataset ends on 2020-07-01, so this is a demonstration on historical
data, not a live forecast for today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.inspect_data import aqi_to_category
from src.preprocessing import StandardScaler, load_processed


def forecast(date: str | None = None, device_name: str = "cpu") -> dict:
    import json

    import torch
    from src.dataset import decode_target
    from src.evaluate import load_variant
    from src.model import pick_device

    if not config.SELECTED_JSON.exists():
        raise FileNotFoundError("No trained models found. Run `python main.py train` first.")
    variant = json.loads(config.SELECTED_JSON.read_text())["selected_variant"]
    device = pick_device(device_name)
    models = load_variant(variant, device)          # one model per seed
    ckpt = models[0][1]
    scaler = StandardScaler(**ckpt["scaler"])
    lookback = ckpt["lookback"]

    clean = load_processed().sort_values("Date").reset_index(drop=True)
    last_date = clean["Date"].max()
    target = pd.Timestamp(date) if date else last_date + pd.Timedelta(days=1)

    history = clean[clean["Date"] < target].tail(lookback)  # strictly BEFORE the target date
    expected_start = target - pd.Timedelta(days=lookback)
    if len(history) < lookback or history["Date"].iloc[0] != expected_start:
        raise ValueError(
            f"Need the {lookback} consecutive days before {target.date()} in the data "
            f"(available: {clean['Date'].min().date()} to {last_date.date()})."
        )

    x = scaler.transform(history)[None, :, :].astype(np.float32)
    today = history[config.TARGET].to_numpy()[-1:]
    per_seed = []
    for model, ck in models:
        with torch.no_grad():
            z = model(torch.from_numpy(x).to(device)).cpu().numpy()
        per_seed.append(float(decode_target(z, today, ck["target_mode"],
                                            (ck["target_mean"], ck["target_std"]))[0]))
    pred = float(np.mean(per_seed))   # average of the 5 runs

    actual_row = clean[clean["Date"] == target]
    actual = float(actual_row[config.TARGET].iloc[0]) if len(actual_row) else None
    return {
        "target_date": target.date().isoformat(),
        "history_from": history["Date"].iloc[0].date().isoformat(),
        "history_to": history["Date"].iloc[-1].date().isoformat(),
        "today_aqi": float(history[config.TARGET].iloc[-1]),
        "variant": variant,
        "predicted_aqi": pred,
        "range_over_runs": (min(per_seed), max(per_seed)),
        "predicted_category": aqi_to_category(max(pred, 0.0)),
        "alert": pred > config.ALERT_THRESHOLD,
        "actual_aqi": actual,
        "actual_category": aqi_to_category(actual) if actual is not None else None,
        "split": actual_row["split"].iloc[0] if len(actual_row) else "beyond dataset",
    }


def print_forecast(r: dict) -> None:
    print("=" * 60)
    print(f" Kolkata next-day AQI forecast for {r['target_date']}")
    print("=" * 60)
    print(f" Based on data from {r['history_from']} to {r['history_to']}")
    print(f" AQI on {r['history_to']} (last known day): {r['today_aqi']:.0f}")
    lo, hi = r["range_over_runs"]
    print(f" Predicted AQI: {r['predicted_aqi']:.0f}  ({r['predicted_category']})"
          f"  [LSTM {r['variant']}, average of 5 runs; runs ranged {lo:.0f}-{hi:.0f}]")
    if r["alert"]:
        print(f" *** HIGH-POLLUTION ALERT: predicted AQI above {config.ALERT_THRESHOLD} ***")
    else:
        print(f" No alert (predicted AQI not above {config.ALERT_THRESHOLD})")
    if r["actual_aqi"] is not None:
        print(f" Actual AQI that day: {r['actual_aqi']:.0f}  ({r['actual_category']}) "
              f"| error {r['predicted_aqi'] - r['actual_aqi']:+.0f} | period: {r['split']}")
    else:
        print(" Actual AQI: not in dataset (the file ends on 2020-07-01)")
    print("=" * 60)


if __name__ == "__main__":
    print_forecast(forecast())
