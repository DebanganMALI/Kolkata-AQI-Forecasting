"""
Checks for the live website forecast.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import live_forecast as lf  # noqa: E402

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestLiveAQI(unittest.TestCase):
    def test_city_aqi_is_mean_of_station_maxima(self):
        recs = [{"station": "A", "pollutant_id": "PM10", "avg_value": "64"},
                {"station": "A", "pollutant_id": "NO2", "avg_value": "30"},
                {"station": "A", "pollutant_id": "CO", "avg_value": "80"},
                {"station": "B", "pollutant_id": "PM2.5", "avg_value": "100"},
                {"station": "B", "pollutant_id": "SO2", "avg_value": "9"},
                {"station": "B", "pollutant_id": "OZONE", "avg_value": "NA"},
                {"station": "B", "pollutant_id": "NH3", "avg_value": "5"}]
        self.assertEqual(lf.city_aqi_from_records(recs), (90.0, 2))

    def test_station_without_pm_is_ignored(self):
        recs = [{"station": "A", "pollutant_id": p, "avg_value": "50"} for p in ("NO2", "CO", "SO2")]
        self.assertEqual(lf.city_aqi_from_records(recs), (None, 0))

    def test_inputs_carry_forward_at_most_7_days(self):
        d0 = date(2026, 11, 1)
        hist = {d0.isoformat(): {"aqi": "100"}}
        days = [d0 + timedelta(days=k) for k in range(8)]
        self.assertEqual(lf.aqi_inputs(hist, days), [100.0] * 8)
        self.assertIsNone(lf.aqi_inputs(hist, days + [d0 + timedelta(days=8)]))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestNumpyGRU(unittest.TestCase):
    def test_numpy_gru_matches_pytorch(self):
        from src.live_model import FEATURES, HIDDEN, export_weights, numpy_gru
        from src.model import AQIGRU
        torch.manual_seed(0)
        m = AQIGRU(len(FEATURES), hidden_size=HIDDEN, num_layers=1).eval()
        x = torch.randn(4, 14, len(FEATURES))
        with torch.no_grad():
            ref = m(x).numpy()
        w = export_weights(m)
        np.testing.assert_allclose(numpy_gru(w, x.numpy().astype(float)), ref, atol=1e-5)
        np.testing.assert_allclose(lf.gru(w, x[0].numpy().astype(float)), ref[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
