"""
backtesting.data_loader — shared OHLCV CSV parsing for the backtest engines.

Extracted so the single-symbol and portfolio engines parse data identically.
Accepts lowercase, Yahoo-Finance, and yfinance multi-header CSV layouts.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

from core.events import BarEvent

_ET = pytz.timezone("America/New_York")

# Ordering weight for interleaving series of different bar sizes at equal
# timestamps (finer bars sort before coarser ones).
_BAR_SECONDS: dict[str, int] = {
    "1 min": 60,
    "5 mins": 300,
    "15 mins": 900,
    "30 mins": 1_800,
    "1 hour": 3_600,
    "4 hours": 14_400,
    "1 day": 86_400,
}


def bar_size_seconds(bar_size: str) -> int:
    """Approximate seconds per bar; used only for stable cross-series ordering."""
    return _BAR_SECONDS.get(bar_size, 86_400)


def _infer_bar_size(rows: list) -> str:
    """Infer bar size from the median interval between consecutive rows."""
    if len(rows) < 2:
        return "1 day"
    deltas = [
        (rows[i + 1][0] - rows[i][0]).total_seconds()
        for i in range(min(len(rows) - 1, 20))
    ]
    median_secs = sorted(deltas)[len(deltas) // 2]
    if median_secs <= 120:
        return "1 min"
    if median_secs <= 1_000:
        return "15 mins"
    if median_secs <= 18_000:        # up to ~5 hours covers 4H bars
        return "4 hours"
    return "1 day"


def parse_ohlcv_csv(
    csv_path: str | Path,
    symbol: Optional[str] = None,
    bar_size: Optional[str] = None,
) -> list[BarEvent]:
    """Parse an OHLCV CSV into a chronologically sorted list of BarEvents.

    Parameters
    ----------
    csv_path : path to the CSV.
    symbol   : ticker; defaults to the file stem upper-cased.
    bar_size : explicit bar size; if None it is inferred from row spacing.
    """
    csv_path = Path(csv_path)
    if symbol is None:
        symbol = csv_path.stem.upper()

    raw_peek = pd.read_csv(csv_path, header=None, nrows=5)

    # yfinance multi-level header: second row holds non-numeric ticker names.
    is_yfinance = False
    if len(raw_peek) >= 2:
        try:
            float(str(raw_peek.iloc[1, 1]))
        except ValueError:
            is_yfinance = True

    if is_yfinance:
        col_names = list(raw_peek.iloc[0])
        df = pd.read_csv(csv_path, skiprows=2, names=col_names)
        rename: dict[str, str] = {col_names[0]: "timestamp"}
        for col in col_names[1:]:
            rename[col] = col.lower()
        df = df.rename(columns=rename)
    else:
        df = pd.read_csv(csv_path)
        col_lower = {c.lower(): c for c in df.columns}
        rename = {}
        for target, candidates in [
            ("timestamp", ["timestamp", "date", "datetime", "time"]),
            ("open",      ["open"]),
            ("high",      ["high"]),
            ("low",       ["low"]),
            ("close",     ["close", "adj close"]),
            ("volume",    ["volume"]),
        ]:
            for cand in candidates:
                if cand in col_lower:
                    rename[col_lower[cand]] = target
                    break
        df = df.rename(columns=rename)

    raw: list[tuple[datetime, float, float, float, float, int]] = []
    for _, row in df.iterrows():
        try:
            ts = pd.to_datetime(str(row["timestamp"]))
        except Exception:
            continue
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            ts = ts.replace(hour=16, minute=0, second=0)   # date-only → close
        ts = _ET.localize(ts) if ts.tzinfo is None else ts.astimezone(_ET)
        raw.append((
            ts.to_pydatetime(),
            float(row["open"]), float(row["high"]),
            float(row["low"]), float(row["close"]),
            int(float(row.get("volume", 0))),
        ))

    raw.sort(key=lambda x: x[0])
    size = bar_size or _infer_bar_size(raw)

    bars = [
        BarEvent(
            symbol=symbol, timestamp=r[0],
            open=Decimal(str(r[1])), high=Decimal(str(r[2])),
            low=Decimal(str(r[3])), close=Decimal(str(r[4])),
            volume=r[5], bar_size=size, vwap=Decimal(str(r[4])),
        )
        for r in raw
    ]

    if not bars:
        raise ValueError(
            f"parse_ohlcv_csv loaded 0 bars from {csv_path}. "
            f"Check the file contains valid OHLCV data.\n"
            f"First rows:\n{raw_peek.to_string(index=False)}"
        )
    return bars
