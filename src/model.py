"""LSTM and GRU forecasters and GPU device setup."""

from __future__ import annotations

import os

import torch
from torch import nn

from src import config


class AQILSTM(nn.Module):
    """LSTM that reads 14 days and forecasts tomorrow's AQI."""
    def __init__(self, n_features: int,
                 hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        """Create the LSTM and output layer."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from the state after the last day."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class AQIGRU(nn.Module):
    """GRU that reads 14 days and forecasts tomorrow's AQI."""

    def __init__(self, n_features: int,
                 hidden_size: int = config.HIDDEN_SIZE,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        """Create the GRU and output layer."""
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from the state after the last day."""
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def pick_device(name: str = config.DEVICE) -> torch.device:
    """Pick the device and make GPU results repeatable."""
    device = _choose_device(name)
    if device.type == "cuda":
        _make_gpu_deterministic()
    return device


def _choose_device(name: str) -> torch.device:
    """Choose CUDA, MPS or CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available - using CPU.")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available (CPU-only PyTorch or no NVIDIA GPU) - using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _make_gpu_deterministic() -> None:
    """Make GPU training give the same result on every run."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_checkpoint(path, device: str = "cpu"):
    """Load a trained model and its settings."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = AQILSTM(
        n_features=len(ckpt["feature_columns"]),
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
