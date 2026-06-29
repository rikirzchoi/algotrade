"""
catalyst_logger — prospective FDA-catalyst data collector (biotech satellite, Path B).

Builds a clean, survivorship-bias-free catalyst dataset by recording events as they
happen (no historical survivor-only data, no look-ahead). v1 logs FDA new-drug
approvals from the free openFDA API; richer catalysts (PDUFA dates, AdCom meetings,
Phase-3 readouts) come from other sources added over time.

Workflow:
  * run periodically (e.g. weekly):  .venv/bin/python -m research.catalyst_logger
  * it appends NEW approvals to research/data/catalysts.csv (deduped by app #)
  * it best-effort-guesses the ticker; unmapped rows get confirmed=0 for you to
    fill in by hand (low volume → manual confirmation is fine and keeps it honest)
  * later, a separate step tracks forward prices for confirmed tickers

This file only COLLECTS. No trading, no edge claim — just an honest growing dataset.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "data" / "catalysts.csv"
FIELDS = ["event_date", "event_type", "application_number",
          "sponsor", "brand", "ticker", "confirmed", "notes"]

# Best-effort sponsor→ticker hints (extend over time). Big pharma is logged but
# usually filtered out later — the edge is in small single-asset biotechs.
SPONSOR_TICKER = {
    "NOVARTIS": "NVS", "ELI LILLY": "LLY", "PFIZER": "PFE", "MERCK": "MRK",
    "BRISTOL": "BMY", "ASTRAZENECA": "AZN", "GLAXO": "GSK", "BAYER": "BAYRY",
    "AMGEN": "AMGN", "GILEAD": "GILD", "BIOGEN": "BIIB", "VERTEX": "VRTX",
    "REGENERON": "REGN", "MODERNA": "MRNA", "BIONTECH": "BNTX", "JAZZ": "JAZZ",
    "ALNYLAM": "ALNY", "NEUROCRINE": "NBIX", "SAREPTA": "SRPT", "INCYTE": "INCY",
    "EXELIXIS": "EXEL", "HALOZYME": "HALO", "UNITED THERAPEUTICS": "UTHR",
    "ABBVIE": "ABBV", "JOHNSON": "JNJ", "TAKEDA": "TAK", "NOVO": "NVO",
}


def _guess_ticker(sponsor: str) -> str:
    up = sponsor.upper()
    for key, tkr in SPONSOR_TICKER.items():
        if key in up:
            return tkr
    return ""


def fetch_approvals(start: str, end: str) -> list[dict]:
    """Original NDA/BLA approvals from openFDA between start/end (YYYYMMDD)."""
    url = "https://api.fda.gov/drug/drugsfda.json"
    params = {
        "search": (f"submissions.submission_type:ORIG AND "
                   f"submissions.submission_status:AP AND "
                   f"submissions.submission_status_date:[{start} TO {end}]"),
        "limit": 1000,
    }
    r = requests.get(url, params=params, timeout=40)
    r.raise_for_status()
    rows: list[dict] = []
    for rec in r.json().get("results", []):
        appn = rec.get("application_number", "")
        if not appn.startswith(("NDA", "BLA")):     # skip generics (ANDA)
            continue
        # Must be an ORIG approval whose OWN date is in-window. (openFDA's nested
        # AND can match a recent SUPPL while the ORIG approval is decades old, so
        # we re-check the date on the actual ORIG/AP submission here.)
        appr = next((s for s in rec.get("submissions", [])
                     if s.get("submission_type") == "ORIG"
                     and s.get("submission_status") == "AP"
                     and start <= s.get("submission_status_date", "") <= end), None)
        if not appr:
            continue
        sponsor = rec.get("sponsor_name", "")
        brand = (rec.get("products", [{}]) or [{}])[0].get("brand_name", "")
        rows.append({
            "event_date": appr.get("submission_status_date", ""),
            "event_type": "FDA_APPROVAL",
            "application_number": appn,
            "sponsor": sponsor,
            "brand": brand,
            "ticker": _guess_ticker(sponsor),
            "confirmed": 0,
            "notes": "",
        })
    return rows


def _load_existing() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open() as fh:
        return {r["application_number"] for r in csv.DictReader(fh)}


def main(lookback_days: int = 365) -> None:
    end = date.today()
    start = end - timedelta(days=lookback_days)
    print(f"Fetching FDA new-drug approvals {start} → {end} …")
    rows = fetch_approvals(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))

    existing = _load_existing()
    new = [r for r in rows if r["application_number"] not in existing]

    write_header = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for r in sorted(new, key=lambda x: x["event_date"]):
            w.writerow(r)

    mapped = sum(1 for r in new if r["ticker"])
    print(f"  {len(rows)} approvals found, {len(new)} new added "
          f"({mapped} auto-mapped, {len(new)-mapped} need manual ticker).")
    print(f"  dataset: {OUT}  (total rows now {len(existing)+len(new)})")
    print("  Next: fill blank tickers in the CSV (set confirmed=1), then a price"
          " tracker records forward returns for each confirmed catalyst.")


if __name__ == "__main__":
    main()
