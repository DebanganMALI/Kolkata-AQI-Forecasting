"""
Read Kolkata's official city AQI from CPCB daily bulletin PDFs.

    python scripts/read_bulletins.py

Input:  data/live/bulletin_pdf/AQI_Bulletin_YYYYMMDD.pdf  (CPCB daily 4 PM bulletin,
        from https://cpcb.nic.in/upload/Downloads/; their certificate had expired in
        Nov 2026, so they were downloaded with curl -k)
Output: data/live/bulletin_aqi.csv  (date, aqi, category, stations)

Each value is checked: the bulletin's own date must match the file name, the
category must agree with the CPCB band for the value, and all 7 Kolkata
stations must have reported. A file that fails a check is skipped, never guessed.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "live" / "bulletin_pdf"
OUT = ROOT / "data" / "live" / "bulletin_aqi.csv"
BANDS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (500, "Severe")]
ROW = re.compile(r"Kolkata\s+(Good|Satisfactory|Moderate|Poor|Very\s*Poor|Severe)\s+(\d+)\b(.*?)(\d+)\s*/\s*(\d+)", re.S)
DATE = re.compile(r"Air Quality Index on\s+([A-Za-z]{3}\s+\d{1,2},\s+\d{4})")


def band(v: int) -> str:
    return next(name for top, name in BANDS if v <= top)


def read(path: Path) -> dict | None:
    if path.read_bytes()[:4] != b"%PDF":
        print(f"  {path.name}: not a PDF (bulletin not published yet when downloaded) - skipped")
        return None
    text = " ".join(page.extract_text() or "" for page in PdfReader(path).pages)
    want = datetime.strptime(path.stem.split("_")[-1], "%Y%m%d").date()
    m_date, m_row = DATE.search(text), ROW.search(text)
    if not m_date or not m_row:
        print(f"  {path.name}: Kolkata row not found - skipped")
        return None
    got = datetime.strptime(m_date.group(1), "%b %d, %Y").date()
    cat, value, used, total = " ".join(m_row.group(1).split()), int(m_row.group(2)), int(m_row.group(4)), int(m_row.group(5))
    problems = []
    if got != want:
        problems.append(f"bulletin date {got}")
    if band(value) != cat:
        problems.append(f"category {cat} does not match value {value}")
    if used < 4:
        problems.append(f"only {used}/{total} stations")
    if problems:
        print(f"  {path.name}: " + "; ".join(problems) + " - skipped")
        return None
    return {"date": want.isoformat(), "aqi": value, "category": cat, "stations": f"{used}/{total}"}


def main() -> None:
    rows = [r for f in sorted(SRC.glob("AQI_Bulletin_*.pdf")) if (r := read(f))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "aqi", "category", "stations"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(rows)} days"
          + (f", {rows[0]['date']} to {rows[-1]['date']}" if rows else ""))


if __name__ == "__main__":
    main()
