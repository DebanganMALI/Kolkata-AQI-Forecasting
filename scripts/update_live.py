"""Daily live update: new CPCB bulletins, weather, government AQI now, and a rebuilt outlook."""

from __future__ import annotations

import csv
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Make the project and scripts folders importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_outlook
import read_bulletins

# File locations, data sources and weather variables.
PDF_DIR = ROOT / "data" / "live" / "bulletin_pdf"
BULLETIN_CSV = ROOT / "data" / "live" / "bulletin_aqi.csv"
WEATHER_CSV = ROOT / "data" / "live" / "weather_recent.csv"
FIRST_DAY = date(2026, 11, 1)
IST = timezone(timedelta(hours=5, minutes=30))
CPCB = "https://cpcb.nic.in/upload/Downloads/AQI_Bulletin_{}.pdf"
WEATHER_VARS = ("temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_hours,wind_speed_10m_max,wind_speed_10m_mean,wind_direction_10m_dominant,"
                "shortwave_radiation_sum,dew_point_2m_mean,relative_humidity_2m_mean,surface_pressure_mean")
UA = {"User-Agent": "Mozilla/5.0 (kolkata-aqi-forecast)"}


def fetch(url: str, verify: bool = True) -> bytes:
    """Download a URL (certificate check can be skipped for CPCB's expired certificate)."""
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90, context=ctx) as r:
        return r.read()


def download_bulletins() -> int:
    """Download every CPCB bulletin missing since 1 Nov 2026."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(IST)
    last = now.date() if now.hour >= 17 else now.date() - timedelta(days=1)
    got = 0
    d = FIRST_DAY
    while d <= last:
        f = PDF_DIR / f"AQI_Bulletin_{d:%Y%m%d}.pdf"
        if not f.exists() or f.read_bytes()[:4] != b"%PDF":
            try:
                data = fetch(CPCB.format(f"{d:%Y%m%d}"), verify=False)
                if data[:4] == b"%PDF":
                    f.write_bytes(data)
                    got += 1
                    print(f"  downloaded bulletin {d}")
                else:
                    print(f"  bulletin {d}: not a PDF (not published yet?)")
            except Exception as e:
                print(f"  bulletin {d}: {e}")
        d += timedelta(days=1)
    return got


def download_weather() -> None:
    """Download recent real weather and the 16-day forecast from Open-Meteo."""
    q = urllib.parse.urlencode({"latitude": 22.5726, "longitude": 88.3639, "daily": WEATHER_VARS,
                                "timezone": "Asia/Kolkata", "past_days": 31, "forecast_days": 16,
                                "format": "csv"})
    WEATHER_CSV.write_bytes(fetch(f"https://api.open-meteo.com/v1/forecast?{q}"))
    print("  weather updated (31 past days + 16-day forecast)")


# Output file and live feed address.
NOW_JSON = ROOT / "web" / "data" / "now.json"
FEED = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"


def government_now() -> None:
    """Compute the government AQI right now from the live CPCB feed and save it."""
    key = os.environ.get("DATA_GOV_API_KEY")
    if not key:
        print("  DATA_GOV_API_KEY not set - skipped")
        return
    from live_forecast import city_aqi_from_records
    q = urllib.parse.urlencode({"api-key": key, "format": "json", "limit": 500, "filters[city]": "Kolkata"})
    try:
        records = json.loads(fetch(f"{FEED}?{q}"))["records"]
    except Exception as e:
        print(f"  live feed not reachable from here: {e}")
        return
    aqi, n = city_aqi_from_records(records)
    if aqi is None:
        print("  live feed had no usable Kolkata readings")
        return
    stamp = max(r.get("last_update", "") for r in records)
    NOW_JSON.write_text(json.dumps({"aqi": aqi, "stations": n, "feed_time": stamp,
                                    "fetched": datetime.now(IST).isoformat(timespec="minutes")}))
    print(f"  government AQI right now: {aqi:.0f} from {n} stations (feed time {stamp})")


def main() -> None:
    """Run all update steps in order."""
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
    print("4. Government AQI right now (data.gov.in live feed)")
    government_now()
    print("5. Outlook")
    build_outlook.main()


if __name__ == "__main__":
    main()
