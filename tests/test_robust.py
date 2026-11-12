"""
Checks for the walk-forward and spike experiment.

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
from src.robust import fold_table, high_days, spike_weights  # noqa: E402

HAS_DATA = config.V2_STATIONS_DIR.exists() and config.V2_WEATHER_CSV.exists()


class TestRobustHelpers(unittest.TestCase):
    def test_spike_weights_favour_polluted_days(self):
        w = spike_weights([50, 100, 200], 100.0, 2.0)
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertTrue(w[2] > w[1] > w[0])
        np.testing.assert_allclose(spike_weights([50, 100], 75.0, 0.0), [1.0, 1.0])

    def test_high_days_are_the_top_share(self):
        y = np.arange(100.0)
        self.assertEqual(int(high_days(y).sum()), int(100 * config.SPIKE_TOP_SHARE))
        self.assertTrue(high_days(y)[-1] and not high_days(y)[0])

    def test_rounds_move_forward_without_overlap(self):
        for a, b in zip(config.ROBUST_FOLDS, config.ROBUST_FOLDS[1:]):
            self.assertLess(a["train_end"], a["val_end"])
            self.assertLess(a["val_end"], a["test_end"])
            self.assertLess(a["train_end"], b["train_end"])


@unittest.skipUnless(HAS_DATA, "extended data files not present")
class TestFolds(unittest.TestCase):
    def test_fold_split_follows_its_dates(self):
        from src.extended_data import build_extended_table
        table, _ = build_extended_table(save=False)
        f = config.ROBUST_FOLDS[0]
        t = fold_table(table, f)
        d = t.groupby("split")["Date"].agg(["min", "max"])
        self.assertEqual(str(d.loc["train", "max"].date()), f["train_end"])
        self.assertEqual(str(d.loc["val", "max"].date()), f["val_end"])
        self.assertEqual(str(d.loc["test", "max"].date()), f["test_end"])
        self.assertTrue((t["Date"] <= pd.Timestamp(f["test_end"])).all())


if __name__ == "__main__":
    unittest.main()
