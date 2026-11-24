"""
Turn the saved test-period forecasts into web/data/test.json for the website.

    python scripts/build_site_data.py

Every forecast here was made from the 14 days before the target day, by models
that never saw the test period (2025-01-01 to 2026-10-31) during training or
model choice. Sources:
  reports/extended_test_predictions.csv  GRU [pollution+weather] (the chosen model),
                                         quantile 10th/90th percentile, persistence
  reports/robust_test_predictions.csv    ensemble, round "2025-2026"
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "test.json"


def main() -> None:
    ext = pd.read_csv(ROOT / "reports" / "extended_test_predictions.csv")
    rob = pd.read_csv(ROOT / "reports" / "robust_test_predictions.csv")
    rob = rob[rob["date"] >= ext["date"].min()][["date", "Ensemble"]]
    df = ext[["date", "actual_AQI", "persistence", "GRU [pollution+weather]",
              "quantile q10 [pollution]", "quantile q90 [pollution]"]].merge(rob, on="date", how="left")
    df.columns = ["date", "actual", "persistence", "gru", "q10", "q90", "ensemble"]
    rows = [{k: (None if pd.isna(v) else (v if k == "date" else round(float(v), 1)))
             for k, v in r.items()} for r in df.to_dict("records")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"period": [rows[0]["date"], rows[-1]["date"]], "days": rows},
                              separators=(",", ":")))
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(rows)} days, "
          f"{int(df['ensemble'].notna().sum())} with an ensemble forecast.")


if __name__ == "__main__":
    main()
