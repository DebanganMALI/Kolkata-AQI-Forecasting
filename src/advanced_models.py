"""
Advanced neural forecasters (used by src/advanced.py).

All of them read the same input as the main LSTM: 14 days x n features, and
output tomorrow's AQI (in the scaled target units of src/dataset.py).

  AttentionLSTM   An LSTM reads the 14 days. An attention layer then gives each
                  day a weight (the weights add up to 1) and mixes the 14 hidden
                  states with those weights. The weights can be read out, which
                  shows WHICH past days the network relied on.
  TCN             Temporal convolutional network: stacked 1-D convolutions that
                  only look backwards in time ("causal"), with gaps between the
                  taps ("dilation" 1, 2, 4) so a few layers cover all 14 days.
  TransformerForecaster
                  A small Transformer encoder: every day attends to every other
                  day (self-attention); a learned position vector tells it which
                  day is which.

Quantile mode (AttentionLSTM with quantiles=True) outputs three numbers per
sample: the 10th, 50th and 90th percentile of tomorrow's AQI. It is trained with
the pinball loss, and the three outputs are built so they can never cross
(q10 <= q50 <= q90).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src import config


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles=config.QUANTILES) -> torch.Tensor:
    """
    Quantile (pinball) loss. For quantile q and error e = y - pred:
        loss = q * e        if the forecast is too low  (e > 0)
        loss = (q - 1) * e  if the forecast is too high (e < 0)
    Minimising it makes pred the q-th percentile: for q = 0.9 a forecast that
    is too low costs 9 times more than one that is too high.
    """
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)   # (Q,)
    e = y.unsqueeze(-1) - pred                                              # (B, Q)
    return torch.maximum(q * e, (q - 1) * e).mean()


def _ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """(B, 3) raw outputs -> (B, 3) = (q10, q50, q90) with q10 <= q50 <= q90 guaranteed."""
    mid = raw[:, 0]
    low = mid - F.softplus(raw[:, 1])     # softplus(x) > 0, so low is always below mid
    high = mid + F.softplus(raw[:, 2])
    return torch.stack([low, mid, high], dim=1)


class AttentionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = config.DROPOUT, quantiles: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # additive (Bahdanau-style) attention: a small network scores each day
        self.score = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, 1, bias=False))
        self.quantiles = quantiles
        # the head sees the attention-weighted summary AND the most recent day
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(2 * hidden_size, 3 if quantiles else 1))
        self.last_attention = None        # (B, days) weights of the latest forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                                       # (B, days, hidden)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1) # (B, days), each row sums to 1
        context = (weights.unsqueeze(-1) * h).sum(dim=1)          # (B, hidden)
        self.last_attention = weights.detach()
        out = self.head(torch.cat([context, h[:, -1, :]], dim=1))
        return _ordered_quantiles(out) if self.quantiles else out.squeeze(-1)


class CausalBlock(nn.Module):
    """Two dilated causal convolutions + a residual (skip) connection."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation        # pad on the LEFT only = never see the future
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, channels, days)
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.skip(x))


class TCN(nn.Module):
    def __init__(self, n_features: int, channels: int = 32, kernel: int = 3,
                 dilations=(1, 2, 4), dropout: float = config.DROPOUT):
        super().__init__()
        blocks, c_in = [], n_features
        for d in dilations:
            blocks.append(CausalBlock(c_in, channels, kernel, d, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, days, features) -> (B, channels, days); position t only depends on days <= t."""
        return self.blocks(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)[:, :, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4, num_layers: int = 2,
                 dropout: float = config.DROPOUT, lookback: int = config.LOOKBACK):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)  # learned positions
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.embed(x) + self.position[:, : x.shape[1]])
        return self.head(z[:, -1, :]).squeeze(-1)
