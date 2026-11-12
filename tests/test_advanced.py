"""
Checks for the advanced deep-learning experiment.

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
from src.data_loader import load_raw  # noqa: E402
from src.dataset import make_windows  # noqa: E402
from src.preprocessing import build_clean_series, fit_scaler  # noqa: E402

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

N_FEATURES = 14


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestAdvanced(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(5, config.LOOKBACK, N_FEATURES)

    def test_attention_weights_sum_to_one(self):
        from src.advanced_models import AttentionLSTM
        m = AttentionLSTM(N_FEATURES).eval()
        out = m(self.x)
        self.assertEqual(tuple(out.shape), (5,))
        self.assertEqual(tuple(m.last_attention.shape), (5, config.LOOKBACK))
        torch.testing.assert_close(m.last_attention.sum(dim=1), torch.ones(5))

    def test_quantiles_never_cross(self):
        from src.advanced_models import AttentionLSTM
        q = AttentionLSTM(N_FEATURES, quantiles=True).eval()(self.x * 10)
        self.assertEqual(tuple(q.shape), (5, 3))
        self.assertTrue(bool((q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()))

    def test_pinball_loss_on_known_numbers(self):
        from src.advanced_models import pinball_loss
        y = torch.tensor([10.0])
        # forecasts 12 (q10), 8 (q50), 8 (q90): losses 0.9*2, 0.5*2, 0.9*2
        loss = pinball_loss(torch.tensor([[12.0, 8.0, 8.0]]), y, (0.1, 0.5, 0.9))
        self.assertAlmostEqual(loss.item(), (1.8 + 1.0 + 1.8) / 3, places=6)

    def test_tcn_is_causal(self):
        """Changing the most recent day must not change the TCN's view of earlier days."""
        from src.advanced_models import TCN
        m = TCN(N_FEATURES).eval()
        x2 = self.x.clone()
        x2[:, -1, :] += 5.0
        with torch.no_grad():
            a, b = m.features(self.x), m.features(x2)
        torch.testing.assert_close(a[:, :, :-1], b[:, :, :-1])
        self.assertFalse(torch.allclose(a[:, :, -1], b[:, :, -1]))

    def test_transformer_output_shape(self):
        from src.advanced_models import TransformerForecaster
        out = TransformerForecaster(N_FEATURES).eval()(self.x)
        self.assertEqual(tuple(out.shape), (5,))


class TestHorizons(unittest.TestCase):
    def test_horizon_target_is_h_days_after_last_input(self):
        clean, _ = build_clean_series(load_raw())
        w = make_windows(clean, fit_scaler(clean), horizon=3)
        dates = pd.to_datetime(clean["Date"]).to_numpy()
        aqi = clean[config.TARGET].to_numpy()
        for k in [0, 200, len(w) - 1]:
            t = np.where(dates == w.target_dates[k])[0][0]
            self.assertEqual(w.y_true[k], aqi[t])
            self.assertEqual(w.last_aqi[k], aqi[t - 3])   # "today" is 3 days before the target


if __name__ == "__main__":
    unittest.main()
