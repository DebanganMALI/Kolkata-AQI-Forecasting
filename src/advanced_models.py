"""Advanced neural forecasters: attention LSTM, TCN and Transformer, plus the pinball loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src import config


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles=config.QUANTILES) -> torch.Tensor:
    """Quantile (pinball) loss averaged over the three quantiles."""
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def _ordered_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Turn raw outputs into quantiles that never cross."""
    mid = raw[:, 0]
    low = mid - F.softplus(raw[:, 1])
    high = mid + F.softplus(raw[:, 2])
    return torch.stack([low, mid, high], dim=1)


class AttentionLSTM(nn.Module):
    """LSTM that weights all 14 days with additive attention before forecasting."""
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = config.DROPOUT, quantiles: bool = False):
        """Create the LSTM, attention scorer and output layer."""
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.score = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, 1, bias=False))
        self.quantiles = quantiles
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(2 * hidden_size, 3 if quantiles else 1))
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from the attention summary and the last hidden state."""
        h, _ = self.lstm(x)
        weights = torch.softmax(self.score(h).squeeze(-1), dim=1)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        self.last_attention = weights.detach()
        out = self.head(torch.cat([context, h[:, -1, :]], dim=1))
        return _ordered_quantiles(out) if self.quantiles else out.squeeze(-1)


class CausalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float):
        """Create the convolutions, dropout and skip connection."""
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block without looking at future days."""
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.skip(x))


class TCN(nn.Module):
    """Temporal convolutional network with dilations 1, 2 and 4."""
    def __init__(self, n_features: int, channels: int = 32, kernel: int = 3,
                 dilations=(1, 2, 4), dropout: float = config.DROPOUT):
        """Stack the causal blocks and the output layer."""
        super().__init__()
        blocks, c_in = [], n_features
        for d in dilations:
            blocks.append(CausalBlock(c_in, channels, kernel, d, dropout))
            c_in = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional features where each day only sees earlier days."""
        return self.blocks(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from the features of the most recent day."""
        return self.head(self.features(x)[:, :, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    """Transformer encoder over the 14 days with learned positions."""
    def __init__(self, n_features: int, d_model: int = 32, n_heads: int = 4, num_layers: int = 2,
                 dropout: float = config.DROPOUT, lookback: int = config.LOOKBACK):
        """Create the embedding, positions, encoder layers and output layer."""
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from the encoded most recent day."""
        z = self.encoder(self.embed(x) + self.position[:, : x.shape[1]])
        return self.head(z[:, -1, :]).squeeze(-1)
