"""Turns the clean daily table into 14-day history windows with next-day AQI targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import StandardScaler, feature_columns


@dataclass
class Windows:
    """Samples: inputs, targets, last-day AQI, target dates and split labels."""
    X: np.ndarray
    y_true: np.ndarray
    last_aqi: np.ndarray
    target_dates: np.ndarray
    split: np.ndarray

    def subset(self, name: str) -> "Windows":
        """Return only the samples of one split."""
        m = self.split == name
        return Windows(self.X[m], self.y_true[m], self.last_aqi[m],
                       self.target_dates[m], self.split[m])

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.y_true)


def make_windows(clean: pd.DataFrame, scaler: StandardScaler,
                 lookback: int = config.LOOKBACK, horizon: int = config.HORIZON) -> Windows:
    """Build every 14-day sample from a gap-free daily table."""
    clean = clean.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(clean["Date"])
    if (dates.diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("Daily table has missing dates; windows would skip days.")

    features = scaler.transform(clean)
    aqi = clean[config.TARGET].to_numpy(dtype=float)
    split = clean["split"].to_numpy()
    h = horizon

    X, y_true, last, tdates, tsplit = [], [], [], [], []
    for t_last in range(lookback - 1, len(clean) - h):
        target = t_last + h
        X.append(features[t_last - lookback + 1: t_last + 1])
        y_true.append(aqi[target])
        last.append(aqi[t_last])
        tdates.append(dates.iloc[target])
        tsplit.append(split[target])

    return Windows(
        X=np.asarray(X, dtype=np.float32),
        y_true=np.asarray(y_true),
        last_aqi=np.asarray(last),
        target_dates=np.asarray(tdates, dtype="datetime64[ns]"),
        split=np.asarray(tsplit),
    )


def target_stats(w_train: Windows, mode: str, scaler: StandardScaler) -> tuple[float, float]:
    """Mean and spread used to scale the target, from training samples only."""
    if mode == "level":
        i = scaler.columns.index(config.TARGET)
        return float(scaler.mean[i]), float(scaler.std[i])
    if mode == "change":
        d = w_train.y_true - w_train.last_aqi
        return float(d.mean()), float(d.std())
    raise ValueError(f"Unknown target mode {mode!r}")


def encode_target(y_true, last_aqi, mode: str, stats: tuple[float, float]) -> np.ndarray:
    """Scale the target (AQI level or day-to-day change) for training."""
    mean, std = stats
    base = np.asarray(y_true, float) if mode == "level" else np.asarray(y_true, float) - np.asarray(last_aqi, float)
    return ((base - mean) / std).astype(np.float32)


def decode_target(z, last_aqi, mode: str, stats: tuple[float, float]) -> np.ndarray:
    """Turn a network output back into a real AQI forecast."""
    mean, std = stats
    v = np.asarray(z, float) * std + mean
    return v if mode == "level" else np.asarray(last_aqi, float) + v


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = config.SEED):
    """Wrap samples in a PyTorch DataLoader."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def feature_names() -> list[str]:
    """Names of the model inputs."""
    return feature_columns()
