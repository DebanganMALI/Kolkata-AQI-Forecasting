"""
Persistence baseline: tomorrow's AQI = today's AQI.

This is the "do nothing clever" forecast. The LSTM is only worth using if it
beats this on the same days.
"""

from __future__ import annotations

import numpy as np

from src.dataset import Windows


def persistence_predict(w: Windows) -> np.ndarray:
    """Prediction for each sample = the real AQI on its last input day."""
    return w.last_aqi.copy()
