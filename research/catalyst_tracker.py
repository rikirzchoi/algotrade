"""
catalyst_tracker — forward-return tracker for confirmed biotech catalysts.

Reads the confirmed (confirmed == 1) rows from catalysts.csv and records what the
stock did AFTER each FDA approval. This is the measurement half of the prospective
biotech study: catalyst_logger collects events survivorship-clean as they happen;
this file scores them.

Methodology (honest, no look-ahead):
  * Entry = the OPEN of the first trading session strictly AFTER the approval date.
    You cannot trade the instant of an approval (often announced after the close),
    so we enter at the next open. This deliberately forgoes the announcement-day
    pop — `gap_pct` records how much moved before we could get in.
  * Horizons = entry-open → close at +1, +5, +20 trading days.
  * Abnormal return = stock return − XBI return over the same window (strips out
    biotech-sector beta), mirroring the defense study's abnormal-vs-ITA approach.

Output: research/data/catalyst_returns.csv (rewritten each run) + a console summary.
NOTE: with only a handful of confirmed catalysts this is DATA ACCUMULATION, not a
result — the sample is far too small to claim an edge (see TRADING_RULES.md §IV).

Run:  .venv/bin/python -m research.catalyst_tracker
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path(__file__).resolve().parent / "data"
CATALYSTS = DATA / "catalysts.csv"
OUT = DATA / "catalyst_returns.csv"
BENCH = "XBI"                       # SPDR S&P Biotech (equal-weight small/mid biotech)
HORIZONS = [1, 5, 20]
OUT_FIELDS = (["event_date", "ticker", "brand", "entry_date", "gap_pct"]
              + [f"{p}_{h}d" for h in HORIZONS for p in ("ret", "abn")]
              + ["status"])


def _load_confirmed() -> list[dict]:
    if not CATALYSTS.exists():
        return []
    with CATALYSTS.open() as fh:
        return [r for r in csv.DictReader(fh)
                if r.get("confirmed") == "1" and r.get("ticker")]


def _forward(df: pd.DataFrame, edate: pd.Timestamp) -> dict | None:
    """Forward returns for one catalyst from a clean [o, c, bo, bc] frame."""
    after = df.index[df.index > edate]
    if len(after) == 0:
        return None
    entry = after[0]
    pos = df.index.get_loc(entry)
    e_open, b_open = df.at[entry, "o"], df.at[entry, "bo"]

    prev = df.index[df.index <= edate]
    gap = (e_open / df.at[prev[-1], "c"] - 1) if len(prev) else np.nan

    res: dict = {"entry_date": entry.date().isoformat(), "gap_pct": gap}
    complete = True
    for h in HORIZONS:
        j = pos + h
        if j >= len(df.index):
            res[f"ret_{h}d"] = res[f"abn_{h}d"] = np.nan
            complete = False
            continue
        x = df.index[j]
        s_ret = df.at[x, "c"] / e_open - 1
        b_ret = df.at[x, "bc"] / b_open - 1
        res[f"ret_{h}d"] = s_ret
        res[f"abn_{h}d"] = s_ret - b_ret
    res["status"] = "complete" if complete else "pending (not enough forward data)"
    return res


def main() -> None:
    cats = _load_confirmed()
    if not cats:
        print("No confirmed catalysts (confirmed==1) to track yet.")
        return

    for c in cats:
        c["_date"] = pd.Timestamp(c["event_date"])
    tickers = sorted({c["ticker"] for c in cats})
    start = (min(c["_date"] for c in cats) - pd.Timedelta(days=7)).date()
    end = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    print(f"Fetching {tickers} + {BENCH}  {start} → {end.date()} …")
    raw = yf.download(tickers + [BENCH], start=start, end=end,
                      auto_adjust=True, progress=False)
    opens, closes = raw["Open"], raw["Close"]

    results: list[dict] = []
    for c in sorted(cats, key=lambda x: x["_date"]):
        t = c["ticker"]
        if t not in closes.columns or closes[t].dropna().empty:
            results.append({**{k: "" for k in OUT_FIELDS}, "event_date": c["event_date"],
                            "ticker": t, "brand": c["brand"], "status": "no price data"})
            continue
        df = pd.DataFrame({"o": opens[t], "c": closes[t],
                           "bo": opens[BENCH], "bc": closes[BENCH]}).dropna()
        fwd = _forward(df, c["_date"])
        row = {"event_date": c["event_date"], "ticker": t, "brand": c["brand"]}
        row.update(fwd or {"status": "no trading day after approval"})
        results.append(row)

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float)
                            and not pd.isna(r[k]) else r.get(k, "")) for k in OUT_FIELDS})

    _print_summary(results)
    print(f"\n  saved: {OUT}")


def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  BIOTECH CATALYST FORWARD RETURNS — entry next-open, abnormal vs XBI")
    print("=" * 70)
    hdr = f"  {'date':10s} {'tkr':5s} {'gap':>7s}"
    for h in HORIZONS:
        hdr += f" {'abn'+str(h)+'d':>8s}"
    print(hdr)
    for r in results:
        line = f"  {r['event_date']:10s} {r['ticker']:5s}"
        line += f" {r['gap_pct']:>+7.1%}" if isinstance(r.get("gap_pct"), float) and not pd.isna(r["gap_pct"]) else f" {'—':>7s}"
        for h in HORIZONS:
            v = r.get(f"abn_{h}d")
            line += f" {v:>+8.1%}" if isinstance(v, float) and not pd.isna(v) else f" {'—':>8s}"
        print(line)

    print("-" * 70)
    n_done = 0
    for h in HORIZONS:
        vals = [r[f"abn_{h}d"] for r in results
                if isinstance(r.get(f"abn_{h}d"), float) and not pd.isna(r[f"abn_{h}d"])]
        if not vals:
            continue
        n_done = max(n_done, len(vals))
        arr = np.array(vals)
        print(f"  abn {h:2d}d:  mean {arr.mean():+.1%}  median {np.median(arr):+.1%}  "
              f"%pos {(arr > 0).mean():.0%}  n={len(arr)}")
    print(f"\n  ⚠️  n≈{n_done}: DATA ACCUMULATION, not a result. The sample is far below"
          "\n      the ≥~100-event, multi-regime bar in TRADING_RULES.md §IV. Do not"
          "\n      infer an edge from this — it is the dataset growing, nothing more.")


if __name__ == "__main__":
    main()
