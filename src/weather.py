"""
Loading the daily Kolkata weather (Open-Meteo Historical Weather API).

The CSV in data/external/ is only ever READ here, exactly like the Kaggle file.

What happens to it:
  * Open-Meteo puts two lines of location metadata above the table; the code
    checks the timezone is Asia/Kolkata (so each "day" matches the AQI dates)
    and then reads the table underneath.
  * Column names are shortened (e.g. "temperature_2m_mean (°C)" -> "temp_mean").
  * Open-Meteo returns the dominant wind direction twice under two names with
    identical values; the duplicate is checked and dropped.
  * Wind direction is an angle, so 359 degrees and 1 degree are almost the same
    direction. Feeding the raw number would tell the model they are far apart,
    so it is turned into sine and cosine (the same trick as the day of year).
  * Weather values are NEVER filled in. If any day in the modelling period has
    no weather, the code stops with an error.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def wind_direction_to_sin_cos(degrees) -> tuple[np.ndarray, np.ndarray]:
    """Angle in degrees -> (sin, cos), so that 0 and 360 degrees are the same point."""
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    return np.sin(radians), np.cos(radians)


def load_weather(path: Path = config.WEATHER_CSV) -> pd.DataFrame:
    """Read the Open-Meteo CSV and return one row per date with the model's weather inputs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the daily Kolkata weather from Open-Meteo "
            "(see README.md, 'Weather data') and save it there."
        )
    lines = path.read_text(encoding="utf-8").splitlines()

    # Metadata block: first line = names, second line = values.
    meta = dict(zip(lines[0].split(","), lines[1].split(",")))
    if meta.get("timezone") != config.WEATHER_TIMEZONE:
        raise ValueError(
            f"Weather file timezone is {meta.get('timezone')!r}, expected "
            f"{config.WEATHER_TIMEZONE!r}. Re-download with timezone=Asia/Kolkata so the "
            "daily values line up with the AQI dates."
        )

    header = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    df = pd.read_csv(StringIO("\n".join(lines[header:])))
    df.columns = [c.split(" (")[0] for c in df.columns]   # drop the unit suffix

    # Duplicate wind-direction column: keep one, but only if they really agree.
    dup = "winddirection_10m_dominant"
    if dup in df.columns:
        if not df[dup].equals(df["wind_direction_10m_dominant"]):
            raise ValueError("The two wind-direction columns differ - check the weather file.")
        df = df.drop(columns=dup)

    missing = [c for c in config.WEATHER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Weather file is missing columns {missing}. Re-download it with "
                         "the variables listed in README.md.")

    df = df.rename(columns=config.WEATHER_COLUMNS)
    df["Date"] = pd.to_datetime(df["time"], format="%Y-%m-%d")
    df["wind_dir_sin"], df["wind_dir_cos"] = wind_direction_to_sin_cos(df["wind_dir_deg"])

    if df["Date"].duplicated().any():
        raise ValueError("Weather file has duplicate dates.")
    return df[["Date"] + list(config.WEATHER_FEATURES)].sort_values("Date").reset_index(drop=True)


def add_weather(clean: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach each day's weather to the clean AQI table by date (no filling)."""
    merged = clean.merge(weather, on="Date", how="left", validate="one_to_one")
    gaps = merged[list(config.WEATHER_FEATURES)].isna().any(axis=1)
    if gaps.any():
        days = merged.loc[gaps, "Date"].dt.date.tolist()
        raise ValueError(f"No weather for {int(gaps.sum())} day(s) in the modelling period "
                         f"({days[:10]}...). Weather values are never invented.")
    return merged
