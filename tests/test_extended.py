"""
Checks for the extended 2018-2026 dataset.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

HAS_DATA = config.V2_STATIONS_DIR.exists() and config.V2_WEATHER_CSV.exists()


@unittest.skipUnless(HAS_DATA, "extended data files not present")
class TestExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.extended import build_data
        from src.extended_data import build_extended_table
        cls.table, cls.log = build_extended_table(save=False)
        cls.w, cls.scaler = build_data(cls.table, "all")

    def test_target_never_filled(self):
        """A day without official AQI is never a forecast target."""
        missing = set(self.table.loc[self.table["AQI_target"].isna(), "Date"])
        self.assertFalse(any(pd.Timestamp(d) in missing for d in self.w.target_dates))
        self.assertFalse(np.isnan(self.w.y_true).any())

    def test_splits_follow_the_dates(self):
        d = self.table.groupby("split")["Date"].agg(["min", "max"])
        self.assertEqual(str(d.loc["train", "max"].date()), config.V2_TRAIN_END)
        self.assertEqual(str(d.loc["val", "max"].date()), config.V2_VAL_END)
        self.assertLess(d.loc["val", "max"], d.loc["test", "min"])

    def test_scaler_uses_train_only(self):
        i = self.scaler.columns.index("PM2.5")
        train = self.table[self.table["split"] == "train"]
        self.assertAlmostEqual(self.scaler.mean[i], train["PM2.5"].mean(), places=6)

    def test_festival_flag_marks_the_forecast_day(self):
        """kali_puja_next is 1 on the day BEFORE the Kali Puja window starts."""
        t = self.table.set_index("Date")
        before, _ = config.V2_EVENT_WINDOWS["kali_puja"]
        first = pd.Timestamp(config.V2_EVENT_DATES["kali_puja"][1]) - pd.Timedelta(days=before)
        self.assertEqual(t.loc[first - pd.Timedelta(days=1), "kali_puja_next"], 1.0)
        self.assertEqual(t.loc[first - pd.Timedelta(days=2), "kali_puja_next"], 0.0)

    def test_weather_daily_block_is_read(self):
        self.assertFalse(self.table[list(config.WEATHER_FEATURES)].isna().any().any())

    def test_fires_are_never_negative(self):
        self.assertTrue((self.table[config.V2_FIRE_FEATURES] >= 0).all().all())

    def test_learning_curve_keeps_only_recent_training_years(self):
        from src.extended import last_years
        w2, _ = last_years((self.w, self.scaler), 2)
        tr = w2.subset("train")
        cut = pd.Timestamp(config.V2_TRAIN_END) - pd.DateOffset(years=2)
        self.assertTrue((pd.to_datetime(tr.target_dates) > cut).all())
        self.assertEqual(len(w2.subset("val")), len(self.w.subset("val")))


if __name__ == "__main__":
    unittest.main()
