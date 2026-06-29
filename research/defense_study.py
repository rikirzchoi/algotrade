"""
defense_study — Gate-1 event study: do material DoD contract awards predict
post-award stock drift? Survivorship-clean (defense names don't delist), free
data (USAspending action-dated awards + yfinance prices).

Method (event study, no look-ahead):
  * catalyst = a NEW base contract award (Mod="0") above a size threshold,
    timestamped by its Action Date (when the action was recorded ≈ announced)
  * enter at the NEXT day's close (never the announcement day we couldn't trade)
  * measure cumulative ABNORMAL return vs the defense sector ETF (ITA) over
    several horizons — isolating company-specific drift from sector moves
  * aggregate across events: mean CAR, t-stat, % positive; split prime vs pure-play
    to test the materiality thesis (a contract moves a small pure-play, not a giant)

Run:  .venv/bin/python -m research.defense_study
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TXN_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
START, END = "2010-01-01", "2024-12-31"
MIN_AWARD = 10_000_000          # ignore sub-$10M noise
HORIZONS = [5, 10, 20, 40]      # trading days
COST = 0.002                    # round-trip cost for the tradeable view

# ticker → (USAspending recipient keyword, tier)
UNIVERSE = {
    "LMT": ("LOCKHEED MARTIN", "prime"), "RTX": ("RAYTHEON", "prime"),
    "NOC": ("NORTHROP GRUMMAN", "prime"), "GD": ("GENERAL DYNAMICS", "prime"),
    "BA": ("BOEING", "prime"), "LHX": ("L3HARRIS", "prime"),
    "HII": ("HUNTINGTON INGALLS", "prime"),
    "KTOS": ("KRATOS", "pure"), "AVAV": ("AEROVIRONMENT", "pure"),
    "MRCY": ("MERCURY SYSTEMS", "pure"), "CACI": ("CACI", "pure"),
    "LDOS": ("LEIDOS", "pure"), "BAH": ("BOOZ ALLEN", "pure"),
    "CW": ("CURTISS-WRIGHT", "pure"), "SAIC": ("SCIENCE APPLICATIONS", "pure"),
}


def fetch_base_awards(keyword: str) -> list[tuple[str, float]]:
    """Largest NEW base contract awards (Mod='0') for a recipient keyword."""
    body = {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],
            "recipient_search_text": [keyword],
            "time_period": [{"start_date": START, "end_date": END}],
        },
        "fields": ["Action Date", "Transaction Amount", "Mod", "Recipient Name"],
        "limit": 100, "sort": "Transaction Amount", "order": "desc",
    }
    r = requests.post(TXN_URL, json=body, timeout=40)
    r.raise_for_status()
    out = []
    for rec in r.json().get("results", []):
        if str(rec.get("Mod")) != "0":
            continue
        amt = rec.get("Transaction Amount") or 0
        if amt >= MIN_AWARD:
            out.append((rec["Action Date"], float(amt)))
    return out


def car(px: pd.Series, bench: pd.Series, events: list[tuple[str, float]]) -> dict[int, list[float]]:
    """Cumulative abnormal returns (vs bench) at each horizon, entering T+1 close."""
    out: dict[int, list[float]] = {h: [] for h in HORIZONS}
    for action_date, _amt in events:
        d = pd.Timestamp(action_date)
        pos = px.index.searchsorted(d)
        entry = pos + 1                          # next day's close
        if entry >= len(px):
            continue
        e_px, e_dt = px.iloc[entry], px.index[entry]
        b_e = bench.asof(e_dt)
        for h in HORIZONS:
            ex = entry + h
            if ex >= len(px) or pd.isna(b_e) or b_e == 0:
                continue
            stock_ret = px.iloc[ex] / e_px - 1
            bench_ret = bench.asof(px.index[ex]) / b_e - 1
            out[h].append(stock_ret - bench_ret)
    return out


def _summary(label: str, abn: dict[int, list[float]]) -> None:
    print(f"\n  {label}")
    print(f"    {'horizon':>8} {'n':>5} {'meanCAR':>9} {'t-stat':>7} {'%pos':>6} {'net/trade':>10}")
    for h in HORIZONS:
        xs = np.array(abn[h], dtype=float)
        if len(xs) < 5:
            continue
        mean = xs.mean()
        t = mean / (xs.std(ddof=1) / np.sqrt(len(xs))) if xs.std() > 0 else 0.0
        pos = (xs > 0).mean()
        print(f"    {h:>6}d {len(xs):>5} {mean:>8.2%} {t:>7.2f} {pos:>5.0%} {mean-COST:>9.2%}")


def main() -> None:
    print("Fetching DoD base awards per defense name (USAspending)…")
    awards, n_ev = {}, 0
    for tkr, (kw, _tier) in UNIVERSE.items():
        try:
            awards[tkr] = fetch_base_awards(kw)
            n_ev += len(awards[tkr])
        except Exception as e:  # noqa: BLE001
            print(f"  {tkr}: fetch failed ({e})"); awards[tkr] = []
    print(f"  {n_ev} qualifying base awards (≥${MIN_AWARD/1e6:.0f}M) across {len(UNIVERSE)} names")

    print("Fetching prices (yfinance)…")
    tickers = list(UNIVERSE) + ["ITA"]
    px = yf.download(tickers, period="max", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    bench = px["ITA"].dropna()

    pooled = {tier: {h: [] for h in HORIZONS} for tier in ("prime", "pure")}
    allabn = {h: [] for h in HORIZONS}
    for tkr, (_kw, tier) in UNIVERSE.items():
        if tkr not in px or not awards[tkr]:
            continue
        abn = car(px[tkr].dropna(), bench, awards[tkr])
        for h in HORIZONS:
            pooled[tier][h] += abn[h]
            allabn[h] += abn[h]

    print("\n" + "=" * 64)
    print("  DEFENSE POST-AWARD DRIFT — Gate 1 event study")
    print("  (cumulative abnormal return vs ITA after a new contract award)")
    print("=" * 64)
    _summary("ALL defense names", allabn)
    _summary("PRIME (large — award usually immaterial)", pooled["prime"])
    _summary("PURE-PLAY (small — award is material)", pooled["pure"])
    print("\n  Read: edge exists if mean CAR is positive with |t|>2 (ideally >3),")
    print("  net of cost — and the thesis predicts it's stronger for pure-plays.\n")


if __name__ == "__main__":
    main()
