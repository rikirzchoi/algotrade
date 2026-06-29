"""
defense_announcements — the ONE pre-committed defense variant (Gate-1, disciplined).

Hypothesis (committed in advance): defense stocks react to the PRECISE DoD daily
contract announcement (published ~5pm ET) — measurable as the overnight gap and a
short post-announcement drift. This differs from the rejected USAspending study by
using exact announcement timing (not noisy action dates) and ALL announced
contracts (not just the 100 largest base awards).

Method (no look-ahead):
  * scrape defense.gov daily contract announcements; parse company + amount + date
  * map to our defense universe; keep NEW awards (drop modifications)
  * GAP (diagnostic, untradeable): announce-day close → next-day open
  * DRIFT (tradeable): buy next-day OPEN, hold N days, abnormal vs ITA
  * aggregate: mean, t-stat (edge needs |t|>2, ideally >3, net of cost)

Run:  .venv/bin/python -m research.defense_announcements
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from research.defense_study import UNIVERSE

UA = {"User-Agent": "Mozilla/5.0 (research; contact: local)"}
LIST_URL = "https://www.defense.gov/News/Contracts/?Page={}"
ART_URL = "https://www.defense.gov/News/Contracts/Contract/Article/{}/"
PAGES = 25                      # ~250 trading days of announcements (override via argv)
HORIZONS = [1, 5, 10, 20]
COST = 0.002


def _tstat(xs: np.ndarray) -> float:
    return xs.mean() / (xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 and xs.std() > 0 else 0.0


def _clean(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def scrape_events(pages: int = PAGES) -> pd.DataFrame:
    kw_to_tkr = {kw.upper(): t for t, (kw, _) in UNIVERSE.items()}
    rows = []
    for pg in range(1, pages + 1):
        try:
            lst = requests.get(LIST_URL.format(pg), timeout=25, headers=UA).text
        except Exception:
            continue
        ids = sorted(set(re.findall(r"/Contract/Article/(\d+)/", lst)))
        if not ids:                       # reached the end of the archive
            print(f"  (no articles on page {pg}; stopping)")
            break
        for aid in ids:
            try:
                body = _clean(requests.get(ART_URL.format(aid), timeout=25, headers=UA).text)
            except Exception:
                continue
            time.sleep(0.15)
            m = re.search(r"contracts for (\w+ \d{1,2},? \d{4})", body, re.I)
            if not m:
                continue
            try:
                adate = pd.to_datetime(m.group(1)).normalize()
            except Exception:
                continue
            # split into contract sentences; match companies
            for sent in re.split(r"(?<=\.)\s+", body):
                up = sent.upper()
                if "AWARDED" not in up:
                    continue
                if "MODIFICATION" in up or re.search(r"P0\d{4}", up):
                    continue                       # new awards only
                tkr = next((t for kw, t in kw_to_tkr.items() if kw in up), None)
                if not tkr:
                    continue
                amt_m = re.search(r"\$[\d,]+", sent)
                amt = float(amt_m.group(0)[1:].replace(",", "")) if amt_m else np.nan
                rows.append({"date": adate, "ticker": tkr, "amount": amt})
        time.sleep(0.15)
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "ticker"])
    return df


def event_study(events: pd.DataFrame) -> None:
    tickers = sorted(events["ticker"].unique().tolist()) + ["ITA"]
    px = yf.download(tickers, period="max", interval="1d",
                     auto_adjust=True, progress=False)
    opens, closes = px["Open"], px["Close"]
    itac = closes["ITA"].dropna()

    gaps, recs = [], []
    for _, ev in events.iterrows():
        t, d = ev["ticker"], ev["date"]
        if t not in closes:
            continue
        c, o = closes[t].dropna(), opens[t].dropna()
        pos = c.index.searchsorted(d)
        if pos == 0 or pos >= len(c) - 1:
            continue
        day_close = c.iloc[pos] if c.index[pos] == d else c.iloc[pos - 1]
        nxt = pos if c.index[pos] > d else pos + 1   # first trading day after d
        if nxt >= len(o):
            continue
        nopen = o.iloc[nxt]
        if pd.isna(day_close) or pd.isna(nopen) or day_close == 0:
            continue
        gaps.append(nopen / day_close - 1)            # overnight gap (diagnostic)
        b0 = itac.asof(c.index[nxt])
        for h in HORIZONS:
            ex = nxt + h
            if ex >= len(c) or pd.isna(b0) or b0 == 0:
                continue
            abn = (c.iloc[ex] / nopen - 1) - (itac.asof(c.index[ex]) / b0 - 1)
            recs.append({"year": int(d.year), "h": h, "abn": abn})
    df = pd.DataFrame(recs)

    print("\n" + "=" * 64)
    print("  DEFENSE ANNOUNCEMENT-TIMING — Gate 2 (multi-year / out-of-sample)")
    print(f"  {len(events)} announcements, {events['date'].min().date()} → "
          f"{events['date'].max().date()}")
    print("=" * 64)
    g = np.array(gaps)
    if len(g) > 5:
        print(f"\n  Overnight GAP: n={len(g)} mean={g.mean():+.2%} "
              f"t={_tstat(g):.2f} %pos={(g>0).mean():.0%}")

    print(f"\n  POOLED drift (abnormal vs ITA):")
    print(f"    {'horizon':>8} {'n':>5} {'mean':>8} {'t-stat':>7} {'%pos':>6} {'net':>8}")
    for h in HORIZONS:
        xs = df[df.h == h]["abn"].to_numpy()
        if len(xs) < 5:
            continue
        print(f"    {h:>6}d {len(xs):>5} {xs.mean():>7.2%} {_tstat(xs):>7.2f} "
              f"{(xs>0).mean():>5.0%} {xs.mean()-COST:>7.2%}")

    print(f"\n  PER-YEAR (20d drift) — the out-of-sample test:")
    print(f"    {'year':>6} {'n':>5} {'mean':>8} {'t-stat':>7} {'%pos':>6}")
    h0 = 20
    sub = df[df.h == h0]
    for yr, grp in sub.groupby("year"):
        xs = grp["abn"].to_numpy()
        if len(xs) < 5:
            print(f"    {yr:>6} {len(xs):>5}   (too few)")
            continue
        print(f"    {yr:>6} {len(xs):>5} {xs.mean():>7.2%} {_tstat(xs):>7.2f} {(xs>0).mean():>5.0%}")
    print("\n  Robust if the negative drift shows up across MOST years, not just")
    print("  the original 2025-26 window. One-year-only = regime artifact.\n")


def main() -> None:
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else PAGES
    print(f"Scraping defense.gov announcements ({pages} pages)…")
    ev = scrape_events(pages)
    ev["date"] = pd.to_datetime(ev["date"])
    out = "research/data/defense_announcements.csv"
    ev.to_csv(out, index=False)
    print(f"  parsed {len(ev)} new-award events for our universe → {out}")
    if len(ev) >= 20:
        event_study(ev)
    else:
        print("  too few events parsed for a study; increase PAGES.")


if __name__ == "__main__":
    main()
