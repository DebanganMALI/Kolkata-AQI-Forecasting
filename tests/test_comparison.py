"""Checks that the weather data and the model comparison follow the research rules."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.compare import (FAMILIES, candidates, feature_set_columns,
                         seeds_for, tabular_feature_names, tabular_features)
from src.data_loader import load_raw
from src.dataset import make_windows
from src.preprocessing import StandardScaler, build_clean_series
from src.weather import add_weather, load_weather, wind_direction_to_sin_cos

# Skip checks whose libraries are missing.
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestComparison(unittest.TestCase):
    """Checks of the weather data and the comparison."""
    @classmethod
    def setUpClass(cls):
        """Load the clean table and weather once."""
        cls.weather_bytes_before = Path(config.WEATHER_CSV).read_bytes()
        cls.clean, _ = build_clean_series(load_raw())
        cls.weather = load_weather()
        cls.merged = add_weather(cls.clean, cls.weather)
        cls.columns = feature_set_columns(True)
        cls.scaler = StandardScaler.fit(cls.merged[cls.merged["split"] == "train"], cls.columns)
        cls.windows = make_windows(cls.merged, cls.scaler)

    def test_weather_file_unchanged(self):
        """The weather file is unchanged."""
        self.assertEqual(Path(config.WEATHER_CSV).read_bytes(), self.weather_bytes_before)

    def test_weather_covers_every_day_without_filling(self):
        """Weather covers every day without filling."""
        self.assertEqual(len(self.merged), len(self.clean))
        self.assertFalse(self.merged[list(config.WEATHER_FEATURES)].isna().any().any())
        gaps = self.weather["Date"].diff().dropna() != pd.Timedelta(days=1)
        self.assertFalse(gaps.any())

    def test_wind_direction_is_circular(self):
        """0 and 360 degrees give the same inputs."""
        s, c = wind_direction_to_sin_cos([1, 359, 181])
        near = np.hypot(s[0] - s[1], c[0] - c[1])
        far = np.hypot(s[0] - s[2], c[0] - c[2])
        self.assertLess(near, 0.05)
        self.assertGreater(far, 1.9)
        np.testing.assert_allclose(self.weather["wind_dir_sin"] ** 2 +
                                   self.weather["wind_dir_cos"] ** 2, 1.0, atol=1e-9)

    def test_pollution_only_set_has_no_weather(self):
        """The pollution-only set has no weather inputs."""
        self.assertTrue(set(feature_set_columns(False)).isdisjoint(config.WEATHER_FEATURES))
        self.assertEqual(feature_set_columns(True)[:len(feature_set_columns(False))],
                         feature_set_columns(False))

    def test_weather_scaler_uses_train_only(self):
        """The weather scaler uses training rows only."""
        train = self.merged[self.merged["split"] == "train"]
        i = self.scaler.columns.index("temp_mean")
        self.assertAlmostEqual(self.scaler.mean[i], train["temp_mean"].mean(), places=6)

    def test_weather_inputs_only_from_past_days(self):
        """Weather inputs end the day before the forecast day."""
        w = self.windows
        j = self.columns.index("temp_mean")
        dates = pd.to_datetime(self.merged["Date"]).to_numpy()
        temp = self.merged["temp_mean"].to_numpy()
        for k in [0, 200, len(w) - 1]:
            t = np.where(dates == w.target_dates[k])[0][0]
            newest = float(self.scaler.inverse_column(w.X[k, -1, j], "temp_mean"))
            self.assertAlmostEqual(newest, temp[t - 1], places=3)

    def test_tabular_features_come_from_the_window(self):
        """Flattened features come from the window itself."""
        X = self.windows.X
        T = tabular_features(X)
        n = X.shape[2]
        self.assertEqual(T.shape[1], len(tabular_feature_names(self.columns)))
        np.testing.assert_array_equal(T[:, :n], X[:, -1, :])
        np.testing.assert_allclose(T[:, -n:],
                                   X[:, -config.TABULAR_RECENT_DAYS:, :].mean(axis=1))

    def test_candidates_are_fixed_lists(self):
        """Candidate settings are fixed before any run."""
        for family in FAMILIES:
            self.assertGreater(len(candidates(family)), 0)
            self.assertGreater(len(seeds_for(family)), 0)
        self.assertEqual([c["name"] for c in candidates("GRU")], list(config.VARIANTS))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_torch_ridge_matches_textbook_ridge(self):
        """GPU ridge matches scikit-learn's ridge."""
        from sklearn.linear_model import Ridge
        from src.compare import TorchRidge
        X = tabular_features(self.windows.subset("train").X).astype(np.float64)
        y = self.windows.subset("train").y_true
        ours = TorchRidge(alpha=10.0).fit(X, y).predict(X)
        ref = Ridge(alpha=10.0).fit(X, y).predict(X)
        np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_gru_output_shape(self):
        """The GRU gives one forecast per sample."""
        import torch
        from src.model import AQIGRU
        m = AQIGRU(n_features=len(self.columns))
        out = m(torch.zeros(4, config.LOOKBACK, len(self.columns)))
        self.assertEqual(tuple(out.shape), (4,))


if __name__ == "__main__":
    unittest.main()
