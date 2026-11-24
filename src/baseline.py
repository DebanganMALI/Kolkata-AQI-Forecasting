"""Persistence baseline: tomorrow's AQI equals today's AQI."""

from __future__ import annotations

import numpy as np

from src.dataset import Windows


def persistence_predict(w: Windows) -> np.ndarray:
    """Predict each sample as the AQI on its last input day."""
    return w.last_aqi.copy()
