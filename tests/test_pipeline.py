"""Checks that the pipeline follows the project's research rules."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.baseline import persistence_predict
from src.data_loader import load_raw
from src.dataset import decode_target, encode_target, make_windows, target_stats
from src.evaluate import (alert_metrics, block_bootstrap_mae_diff,
                          regression_metrics, tune_threshold)
from src.preprocessing import build_clean_series, feature_columns, fit_scaler

# Skip the PyTorch check when it is missing.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPipeline(unittest.TestCase):
    """Checks of preprocessing, windows, metrics and models."""
    @classmethod
    def setUpClass(cls):
        """Build the clean table and windows once."""
        cls.raw_hash_before = Path(config.RAW_CSV).read_bytes()
        cls.raw = load_raw()
        cls.clean, cls.log = build_clean_series(cls.raw)
        cls.scaler = fit_scaler(cls.clean)
        cls.windows = make_windows(cls.clean, cls.scaler)

    def test_raw_file_unchanged(self):
        """The raw file is unchanged."""
        self.assertEqual(Path(config.RAW_CSV).read_bytes(), self.raw_hash_before)

    def test_target_never_imputed(self):
        """The target is never filled in."""
        self.assertFalse(self.clean[config.TARGET].isna().any())
        self.assertFalse(self.clean[f"{config.TARGET}_imputed"].any())

    def test_imputed_cells_are_flagged(self):
        """Every filled input is flagged."""
        n_flagged = sum(int(self.clean[f"{c}_imputed"].sum()) for c in config.POLLUTANT_FEATURES)
        self.assertEqual(n_flagged, self.log["n_cells_filled"])

    def test_forward_fill_uses_previous_day(self):
        """Filling uses the previous day."""
        for col in config.POLLUTANT_FEATURES:
            idx = np.where(self.clean[f"{col}_imputed"].to_numpy())[0]
            for i in idx:
                self.assertEqual(self.clean[col].iloc[i], self.clean[col].iloc[i - 1])

    def test_no_xylene(self):
        """Xylene is excluded."""
        self.assertNotIn("Xylene", feature_columns())

    def test_splits_are_chronological_and_disjoint(self):
        """Splits are chronological and do not overlap."""
        d = self.clean.groupby("split")["Date"].agg(["min", "max"])
        self.assertLess(d.loc["train", "max"], d.loc["val", "min"])
        self.assertLess(d.loc["val", "max"], d.loc["test", "min"])

    def test_scaler_uses_train_only(self):
        """The scaler uses training rows only."""
        train = self.clean[self.clean["split"] == "train"]
        i = self.scaler.columns.index(config.TARGET)
        self.assertAlmostEqual(self.scaler.mean[i], train[config.TARGET].mean(), places=6)
        self.assertNotAlmostEqual(self.scaler.mean[i], self.clean[config.TARGET].mean(), places=3)

    def test_windows_use_only_past_days(self):
        """Each window ends the day before its target."""
        w = self.windows
        aqi_scaled_col = feature_columns().index(config.TARGET)
        dates = pd.to_datetime(self.clean["Date"]).to_numpy()
        aqi = self.clean[config.TARGET].to_numpy()
        for k in [0, 100, len(w) - 1]:
            t = np.where(dates == w.target_dates[k])[0][0]
            self.assertEqual(w.y_true[k], aqi[t])
            last_input = self.scaler.inverse_column(w.X[k, -1, aqi_scaled_col], config.TARGET)
            self.assertAlmostEqual(float(last_input), aqi[t - 1], places=3)
            self.assertEqual(w.X.shape[1], config.LOOKBACK)

    def test_every_sample_target_in_its_split(self):
        """Each sample's target is in its own split."""
        split_of_date = dict(zip(pd.to_datetime(self.clean["Date"]).to_numpy(), self.clean["split"]))
        for d, s in zip(self.windows.target_dates, self.windows.split):
            self.assertEqual(split_of_date[d], s)

    def test_persistence_is_previous_day(self):
        """Persistence equals the previous day's AQI."""
        w = self.windows.subset("test")
        np.testing.assert_array_equal(persistence_predict(w), w.last_aqi)

    def test_metrics_on_known_numbers(self):
        """Metrics match hand calculations."""
        m = regression_metrics(np.array([100, 200, 300]), np.array([110, 190, 300]))
        self.assertAlmostEqual(m["MAE"], 20 / 3)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(200 / 3))
        a = alert_metrics(np.array([150, 250, 250, 100]), np.array([210, 260, 150, 90]), 200)
        self.assertEqual((a["correct_alerts_TP"], a["false_alarms_FP"], a["missed_FN"]), (1, 1, 1))
        self.assertAlmostEqual(a["precision"], 0.5)
        self.assertAlmostEqual(a["recall"], 0.5)

    def test_target_encoding_round_trip(self):
        """Encoding and decoding the target round-trips."""
        w = self.windows.subset("train")
        for mode in ("level", "change"):
            stats = target_stats(w, mode, self.scaler)
            z = encode_target(w.y_true, w.last_aqi, mode, stats)
            back = decode_target(z, w.last_aqi, mode, stats)
            np.testing.assert_allclose(back, w.y_true, atol=1e-3)

    def test_change_mode_zero_output_equals_persistence(self):
        """In change mode, zero output gives persistence."""
        w = self.windows.subset("test")
        stats = target_stats(self.windows.subset("train"), "change", self.scaler)
        z_no_change = np.full(len(w), (0 - stats[0]) / stats[1])
        np.testing.assert_allclose(decode_target(z_no_change, w.last_aqi, "change", stats),
                                   persistence_predict(w), atol=1e-6)

    def test_change_stats_use_train_only(self):
        """Change statistics use training samples only."""
        tr = self.windows.subset("train")
        stats = target_stats(tr, "change", self.scaler)
        self.assertAlmostEqual(stats[0], float((tr.y_true - tr.last_aqi).mean()), places=6)

    def test_threshold_tuning_and_bootstrap(self):
        """Threshold tuning and bootstrap behave as expected."""
        y = np.array([100, 250, 260, 120, 300, 90])
        p = np.array([110, 190, 255, 130, 280, 95])
        self.assertEqual(tune_threshold(y, p), 185)
        d, lo, hi = block_bootstrap_mae_diff(y, p, p)
        self.assertEqual((d, lo, hi), (0.0, 0.0, 0.0))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_model_output_shape(self):
        """The LSTM gives one forecast per sample."""
        import torch
        from src.model import AQILSTM
        m = AQILSTM(n_features=len(feature_columns()))
        out = m(torch.zeros(5, config.LOOKBACK, len(feature_columns())))
        self.assertEqual(tuple(out.shape), (5,))


if __name__ == "__main__":
    unittest.main()
