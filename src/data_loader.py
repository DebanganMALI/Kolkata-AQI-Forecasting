"""
Loading the raw Kaggle file.

The raw CSV is only ever READ here. Nothing in the project writes to
data/raw/, so the original download stays exactly as it was.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Read city_day.csv exactly as downloaded."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Kaggle file city_day.csv in data/raw/."
        )
    return pd.read_csv(path)


def load_city(raw: pd.DataFrame, city: str = config.CITY) -> pd.DataFrame:
    """Return one city's rows, with Date parsed and sorted oldest -> newest."""
    df = raw[raw["City"] == city].copy()
    if df.empty:
        raise ValueError(f"No rows found for City == {city!r}")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values("Date").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError(f"{city} has duplicate dates - check the raw file.")
    return df
