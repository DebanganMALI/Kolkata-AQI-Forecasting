"""Turns the saved test-period forecasts and weather into web/data/test.json for the website."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Project folder and output file.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "test.json"


def main() -> None:
    """Merge GRU, ensemble, persistence, forecast range and weather per test day and save JSON."""
    ext = pd.read_csv(ROOT / "reports" / "extended_test_predictions.csv")
    rob = pd.read_csv(ROOT / "reports" / "robust_test_predictions.csv")
    rob = rob[rob["date"] >= ext["date"].min()][["date", "Ensemble"]]
    df = ext[["date", "actual_AQI", "persistence", "GRU [pollution+weather]",
              "quantile q10 [pollution]", "quantile q90 [pollution]"]].merge(rob, on="date", how="left")
    df.columns = ["date", "actual", "persistence", "gru", "q10", "q90", "ensemble"]
    w = pd.read_csv(ROOT / "data" / "processed" / "kolkata_extended_daily.csv")
    w = w[["Date", "temp_mean", "temp_max", "temp_min", "rain_mm", "humidity", "wind_mean",
           "wind_dir_sin", "wind_dir_cos"]].rename(columns={"Date": "date"})
    w["wind_dir"] = (np.degrees(np.arctan2(w.pop("wind_dir_sin"), w.pop("wind_dir_cos"))) % 360).round()
    df = df.merge(w, on="date", how="left")
    rows = [{k: (None if pd.isna(v) else (v if k == "date" else round(float(v), 1)))
             for k, v in r.items()} for r in df.to_dict("records")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"period": [rows[0]["date"], rows[-1]["date"]], "days": rows},
                              separators=(",", ":")))
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(rows)} days, "
          f"{int(df['ensemble'].notna().sum())} with an ensemble forecast.")


if __name__ == "__main__":
    main()
