"""
Daily live update for the website (run by GitHub Actions at 17:45 IST, or by hand).

    python scripts/update_live.py

1. Downloads every CPCB daily AQI bulletin (4 PM, 24-hour average) that is
   missing since the last one saved, up to today.
2. Reads Kolkata's official city AQI from them (scripts/read_bulletins.py).
3. Downloads recent real weather + the 16-day forecast from Open-Meteo.
4. Rebuilds the outlook (scripts/build_outlook.py): the newest official value
   becomes the starting point, so "tomorrow" is always a one-day-ahead forecast.

Note on cpcb.nic.in: its HTTPS certificate had expired (Nov 2026), so the
bulletins are fetched without certificate checking, for that host only. They
are public PDFs that are only read, and every value is checked (date, category,
station count) before use.
"""

from __future__ import annotations

import csv
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_outlook  # noqa: E402
import read_bulletins  # noqa: E402

PDF_DIR = ROOT / "data" / "live" / "bulletin_pdf"
BULLETIN_CSV = ROOT / "data" / "live" / "bulletin_aqi.csv"
WEATHER_CSV = ROOT / "data" / "live" / "weather_recent.csv"
FIRST_DAY = date(2026, 11, 1)                 # the main data ends on 31 Oct 2026
IST = timezone(timedelta(hours=5, minutes=30))
CPCB = "https://cpcb.nic.in/upload/Downloads/AQI_Bulletin_{}.pdf"
WEATHER_VARS = ("temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_hours,wind_speed_10m_max,wind_speed_10m_mean,wind_direction_10m_dominant,"
                "shortwave_radiation_sum,dew_point_2m_mean,relative_humidity_2m_mean,surface_pressure_mean")
UA = {"User-Agent": "Mozilla/5.0 (kolkata-aqi-forecast)"}


def fetch(url: str, verify: bool = True) -> bytes:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90, context=ctx) as r:
        return r.read()


def download_bulletins() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(IST)
    last = now.date() if now.hour >= 17 else now.date() - timedelta(days=1)   # today's appears after ~4:30 PM
    got = 0
    d = FIRST_DAY
    while d <= last:
        f = PDF_DIR / f"AQI_Bulletin_{d:%Y%m%d}.pdf"
        if not f.exists() or f.read_bytes()[:4] != b"%PDF":     # missing, or saved before it was published
            try:
                data = fetch(CPCB.format(f"{d:%Y%m%d}"), verify=False)
                if data[:4] == b"%PDF":
                    f.write_bytes(data)
                    got += 1
                    print(f"  downloaded bulletin {d}")
                else:
                    print(f"  bulletin {d}: not a PDF (not published yet?)")
            except Exception as e:                     # noqa: BLE001
                print(f"  bulletin {d}: {e}")
        d += timedelta(days=1)
    return got


def download_weather() -> None:
    q = urllib.parse.urlencode({"latitude": 22.5726, "longitude": 88.3639, "daily": WEATHER_VARS,
                                "timezone": "Asia/Kolkata", "past_days": 31, "forecast_days": 16,
                                "format": "csv"})
    WEATHER_CSV.write_bytes(fetch(f"https://api.open-meteo.com/v1/forecast?{q}"))
    print("  weather updated (31 past days + 16-day forecast)")


def main() -> None:
    print("1. CPCB bulletins")
    download_bulletins()
    print("2. Reading Kolkata values")
    read_bulletins.main()
    with BULLETIN_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    if rows:
        print(f"   latest official AQI: {rows[-1]['aqi']} ({rows[-1]['category']}) on {rows[-1]['date']}")
    print("3. Weather")
    download_weather()
    print("4. Outlook")
    build_outlook.main()


if __name__ == "__main__":
    main()
