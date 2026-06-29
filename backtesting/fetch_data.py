"""
fetch_data — download historical OHLCV from Yahoo Finance into data/ for
backtesting. Saves clean `timestamp,open,high,low,close,volume` CSVs that
backtesting.data_loader.parse_ohlcv_csv reads.

Run:  .venv/bin/python -m backtesting.fetch_data

IMPORTANT data limits (Yahoo, free): 1m ≈ 7d, 5m/15m ≈ 60d, 1h ≈ 730d, 1d ≈ full
history. So only daily strategies get a multi-year (Gate-1-grade) sample; intraday
strategies are limited to ~60 days here and need a paid/IBKR source for rigorous
validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger("fetch_data")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (symbol, yahoo_interval, period, resample_to) — resample_to None = keep as-is
FETCH_PLAN = [
    ("QQQ",  "1d",  "10y", None),     # momentum_breakout
    ("SPY",  "1d",  "10y", None),     # benchmark for attribution
    ("GLD",  "1h",  "730d", "4h"),    # trend_following (resampled 1h → 4h)
    ("USO",  "1h",  "730d", "4h"),    # trend_following
    ("MSFT", "15m", "60d", None),     # bollinger_reversion
    ("NVDA", "15m", "60d", None),     # bollinger_reversion
    ("AMZN", "15m", "60d", None),     # bollinger_reversion
    ("AMZN", "5m",  "60d", None),     # opening_range_breakout
]

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex columns and lower-case them."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_one(symbol: str, interval: str, period: str, resample_to: str | None) -> Path:
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"no data for {symbol} {interval}")
    df = _flatten(df)
    if resample_to:
        df = df.resample(resample_to).agg(_AGG).dropna()
        interval = resample_to
    df.index.name = "timestamp"

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{symbol}_{interval}.csv"
    df.to_csv(out)
    log.info("saved %-22s %5d bars  %s → %s", out.name, len(df),
             str(df.index[0])[:10], str(df.index[-1])[:10])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Downloading to {DATA_DIR}/ ...\n")
    for symbol, interval, period, resample_to in FETCH_PLAN:
        try:
            fetch_one(symbol, interval, period, resample_to)
        except Exception as exc:  # noqa: BLE001
            log.warning("FAILED %s %s: %s", symbol, interval, exc)
    print("\nDone.")


if __name__ == "__main__":
    main()
