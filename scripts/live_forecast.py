"""Helpers for the live forecast: city AQI from the CPCB feed, AQI history files and the NumPy GRU."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# File locations.
ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
AQI_CSV = LIVE / "aqi_daily.csv"
FORECASTS_CSV = LIVE / "forecasts.csv"
WEIGHTS = ROOT / "models" / "live" / "gru_live.npz"
META = ROOT / "models" / "live" / "gru_live.json"
SITE = ROOT / "web" / "data" / "live.json"

# Data source, coordinates, time zone and weather variables.
RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
LAT, LON = 22.5726, 88.3639
IST = timezone(timedelta(hours=5, minutes=30))
FFILL_LIMIT = 7
WEATHER_DAILY = {
    "temperature_2m_mean": "temp_mean", "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min", "precipitation_sum": "rain_mm",
    "precipitation_hours": "rain_hours", "wind_speed_10m_max": "wind_max",
    "wind_speed_10m_mean": "wind_mean", "wind_direction_10m_dominant": "wind_dir_deg",
    "shortwave_radiation_sum": "radiation", "dew_point_2m_mean": "dew_point",
    "relative_humidity_2m_mean": "humidity", "surface_pressure_mean": "pressure",
}
CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
              (300, "Poor"), (400, "Very Poor"), (10_000, "Severe")]


def category(aqi: float) -> str:
    """Return the CPCB category name for an AQI value."""
    return next(name for top, name in CATEGORIES if aqi <= top)


def get_json(url: str) -> dict:
    """Download a URL and parse it as JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": "kolkata-aqi-forecast"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def read_aqi() -> dict[str, dict]:
    """Read the saved daily AQI history."""
    if not AQI_CSV.exists():
        return {}
    with AQI_CSV.open(newline="") as f:
        return {r["date"]: r for r in csv.DictReader(f)}


def write_aqi(rows: dict[str, dict]) -> None:
    """Save the daily AQI history."""
    LIVE.mkdir(parents=True, exist_ok=True)
    with AQI_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "aqi", "source", "stations"])
        w.writeheader()
        for d in sorted(rows):
            w.writerow(rows[d])


def city_aqi_from_records(records: list[dict]) -> tuple[float | None, int]:
    """City AQI = mean over stations of each station's highest pollutant sub-index."""
    by_station: dict[str, dict[str, float]] = {}
    for r in records:
        try:
            v = float(r.get("avg_value"))
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        by_station.setdefault(r["station"], {})[r["pollutant_id"].upper()] = v
    station_aqi = [max(p.values()) for p in by_station.values()
                   if len(p) >= 3 and ("PM2.5" in p or "PM10" in p)]
    if not station_aqi:
        return None, 0
    return float(round(sum(station_aqi) / len(station_aqi))), len(station_aqi)


def fetch_live_aqi(key: str) -> tuple[float | None, int, str]:
    """Fetch the live CPCB feed for Kolkata and compute the city AQI."""
    q = urllib.parse.urlencode({"api-key": key, "format": "json", "limit": 500,
                                "filters[city]": "Kolkata"})
    records = get_json(f"https://api.data.gov.in/resource/{RESOURCE}?{q}").get("records", [])
    aqi, n = city_aqi_from_records(records)
    stamps = sorted(r.get("last_update", "") for r in records)
    return aqi, n, (stamps[-1] if stamps else "")


def fetch_weather(start: date, end: date) -> dict[str, dict]:
    """Download recent daily weather from Open-Meteo."""
    q = urllib.parse.urlencode({
        "latitude": LAT, "longitude": LON, "timezone": "Asia/Kolkata",
        "daily": ",".join(WEATHER_DAILY), "past_days": (date.today() - start).days + 2,
        "forecast_days": 2})
    d = get_json(f"https://api.open-meteo.com/v1/forecast?{q}")["daily"]
    out = {}
    for i, day in enumerate(d["time"]):
        if start.isoformat() <= day <= end.isoformat():
            row = {short: d[long][i] for long, short in WEATHER_DAILY.items()}
            if any(v is None for v in row.values()):
                raise RuntimeError(f"Open-Meteo has no complete weather for {day}")
            ang = math.radians(row.pop("wind_dir_deg"))
            row["wind_dir_sin"], row["wind_dir_cos"] = math.sin(ang), math.cos(ang)
            out[day] = row
    return out


def load_model():
    """Load the live GRU settings and weights."""
    meta = json.loads(META.read_text())
    raw = np.load(WEIGHTS)
    seeds = [{k: raw[f"s{i}_{k}"] for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh",
                                             "head_w", "head_b")} for i in range(meta["seeds"])]
    return meta, seeds


def gru(w: dict, x: np.ndarray) -> float:
    """Run the GRU and its output layer in NumPy for one 14-day window."""
    H = w["weight_hh"].shape[1]
    h = np.zeros(H)
    sig = lambda a: 1.0 / (1.0 + np.exp(-a))
    for step in x:
        gi, gh = w["weight_ih"] @ step + w["bias_ih"], w["weight_hh"] @ h + w["bias_hh"]
        r, z = sig(gi[:H] + gh[:H]), sig(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1 - z) * n + z * h
    return float(w["head_w"][0] @ h + w["head_b"][0])


def aqi_inputs(history: dict[str, dict], days: list[date]) -> list[float] | None:
    """Daily AQI inputs, carried forward at most 7 days; None if a gap is longer."""
    known = {d: float(r["aqi"]) for d, r in history.items() if r["aqi"] not in ("", None)}
    out = []
    for day in days:
        for back in range(FFILL_LIMIT + 1):
            key = (day - timedelta(days=back)).isoformat()
            if key in known:
                out.append(known[key])
                break
        else:
            return None
    return out


def forecast(today: date, history: dict[str, dict]) -> dict | None:
    """Forecast tomorrow's AQI from the last 14 days."""
    meta, seeds = load_model()
    days = [today - timedelta(days=k) for k in range(meta["lookback"] - 1, -1, -1)]
    aqi = aqi_inputs(history, days)
    if aqi is None:
        print("Not enough AQI history for a forecast (gap longer than 7 days).")
        return None
    weather = fetch_weather(days[0], days[-1])
    rows = []
    for day, a in zip(days, aqi):
        doy = day.timetuple().tm_yday
        n_days = 366 if (day.year % 4 == 0 and (day.year % 100 != 0 or day.year % 400 == 0)) else 365
        ang = 2 * math.pi * (doy - 1) / n_days
        feats = {"AQI": a, **weather[day.isoformat()], "doy_sin": math.sin(ang), "doy_cos": math.cos(ang)}
        rows.append([feats[f] for f in meta["features"]])
    x = (np.array(rows, float) - np.array(meta["feature_mean"])) / np.array(meta["feature_std"])
    s, last = meta["target"], aqi[-1]
    preds = []
    for w in seeds:
        z = gru(w, x)
        preds.append(z * s["std"] + s["mean"] if meta["mode"] == "level" else last + z * s["dstd"])
    p = float(np.mean(preds))
    lo, hi = meta["range80"]
    return {"made_on": today.isoformat(), "target_date": (today + timedelta(days=1)).isoformat(),
            "forecast": round(p), "low": max(0, round(p + lo)), "high": round(p + hi)}


def read_forecasts() -> dict[str, dict]:
    """Read the saved forecasts."""
    if not FORECASTS_CSV.exists():
        return {}
    with FORECASTS_CSV.open(newline="") as f:
        return {r["target_date"]: r for r in csv.DictReader(f)}


def write_forecasts(rows: dict[str, dict]) -> None:
    """Save the forecasts."""
    with FORECASTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["made_on", "target_date", "forecast", "low", "high"])
        w.writeheader()
        for d in sorted(rows):
            w.writerow(rows[d])


def write_site(history, forecasts, feed_time) -> None:
    """Write the live forecast data file for the website."""
    track = []
    for d in sorted(forecasts):
        f = forecasts[d]
        actual = history.get(d, {}).get("aqi", "")
        track.append({"date": d, "forecast": float(f["forecast"]), "low": float(f["low"]),
                      "high": float(f["high"]),
                      "actual": float(actual) if actual not in ("", None) else None})
    latest = forecasts[max(forecasts)] if forecasts else None
    recent = [{"date": d, "aqi": float(history[d]["aqi"]), "source": history[d]["source"]}
              for d in sorted(history)[-30:] if history[d]["aqi"] not in ("", None)]
    site = {"updated": datetime.now(IST).isoformat(timespec="minutes"), "feed_time": feed_time,
            "tomorrow": None if latest is None else {
                "date": latest["target_date"], "aqi": float(latest["forecast"]),
                "low": float(latest["low"]), "high": float(latest["high"]),
                "category": category(float(latest["forecast"]))},
            "track": track, "recent_aqi": recent}
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(json.dumps(site, indent=2))


def daily() -> None:
    """Daily run: save today's AQI, forecast tomorrow and update the site data."""
    key = os.environ.get("DATA_GOV_API_KEY")
    if not key:
        sys.exit("Set DATA_GOV_API_KEY (data.gov.in API key).")
    today = datetime.now(IST).date()
    history = read_aqi()
    aqi, n, feed_time = fetch_live_aqi(key)
    if aqi is None:
        print("The live feed returned no usable Kolkata readings today; no AQI saved.")
    else:
        history[today.isoformat()] = {"date": today.isoformat(), "aqi": aqi,
                                      "source": "cpcb-live", "stations": n}
        print(f"{today}: city AQI {aqi:.0f} from {n} stations (feed time {feed_time})")
    write_aqi(history)

    forecasts = read_forecasts()
    f = forecast(today, history)
    if f:
        forecasts[f["target_date"]] = f
        print(f"Forecast for {f['target_date']}: {f['forecast']} (80% range {f['low']}-{f['high']})")
    write_forecasts(forecasts)
    write_site(history, forecasts, feed_time)


def seed() -> None:
    """Fill the AQI history from the official CPCB city bulletin files."""
    sys.path.insert(0, str(ROOT))
    from src.extended_data import convert_city_aqi_xlsx
    src_dir = LIVE / "bulletin"
    if not list(src_dir.glob("AQI_Data_City_Level_Kolkata_*_Daily.xlsx")):
        sys.exit(f"Save the newest CPCB city AQI file (AQI_Data_City_Level_Kolkata_2026_Daily.xlsx) in {src_dir}")
    tmp = LIVE / "_bulletin.csv"
    df = convert_city_aqi_xlsx(src_dir, tmp)
    tmp.unlink()
    df = df.dropna(subset=["AQI"])
    df = df[df["Date"] >= (date.today() - timedelta(days=60)).isoformat()]
    history = read_aqi()
    for d, a in zip(df["Date"], df["AQI"]):
        if history.get(d, {}).get("source") != "cpcb-live":
            history[d] = {"date": d, "aqi": float(a), "source": "cpcb-bulletin", "stations": ""}
    write_aqi(history)
    print(f"History now has {len(history)} days, last {max(history)}.")


if __name__ == "__main__":
    seed() if sys.argv[1:] == ["seed"] else daily()
