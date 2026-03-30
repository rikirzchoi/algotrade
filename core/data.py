"""
core.data — bar history, technical indicators, and market-structure helpers.

Provides BarHistory for storing and querying BarEvent sequences, plus
standalone analytical functions (SMA, EMA, RSI, Bollinger Bands, VWAP,
volume profile, opening range, n-day high/low, time-to-flat, etc.) that
strategies consume.

This module never imports ibapi and never touches the broker.  All
calculations are parameter-driven — no AppConfig imports.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque

import numpy as np
import pandas as pd
import pytz

from core.events import BarEvent

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# BarHistory
# ---------------------------------------------------------------------------

class BarHistory:
    """Rolling window of BarEvent objects for one symbol.

    Supports slicing, array-view properties (closes, highs, …), and
    convenience queries such as bars_today() and since().
    """

    def __init__(self, symbol: str, maxlen: int = 500) -> None:
        """Initialise an empty history for *symbol* with a rolling *maxlen*."""
        self.symbol: str = symbol
        self._bars: Deque[BarEvent] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Core container protocol
    # ------------------------------------------------------------------

    def append(self, bar: BarEvent) -> None:
        """Append *bar*, attaching ET timezone if the timestamp is naive."""
        if bar.timestamp.tzinfo is None:
            ts = _ET.localize(bar.timestamp)
            # BarEvent is frozen so we rebuild with the tz-aware timestamp
            bar = BarEvent(
                symbol=bar.symbol,
                timestamp=ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                bar_size=bar.bar_size,
                vwap=bar.vwap,
            )
        self._bars.append(bar)

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index: int) -> BarEvent:
        """Support positive and negative indexing; bars[-1] is the most recent."""
        return list(self._bars)[index]

    # ------------------------------------------------------------------
    # Array views
    # ------------------------------------------------------------------

    @property
    def closes(self) -> np.ndarray:
        """Float array of close prices, oldest first."""
        return np.array([float(b.close) for b in self._bars], dtype=float)

    @property
    def highs(self) -> np.ndarray:
        """Float array of high prices, oldest first."""
        return np.array([float(b.high) for b in self._bars], dtype=float)

    @property
    def lows(self) -> np.ndarray:
        """Float array of low prices, oldest first."""
        return np.array([float(b.low) for b in self._bars], dtype=float)

    @property
    def volumes(self) -> np.ndarray:
        """Integer array of volumes, oldest first."""
        return np.array([b.volume for b in self._bars], dtype=float)

    @property
    def timestamps(self) -> list[datetime]:
        """List of timestamps, oldest first."""
        return [b.timestamp for b in self._bars]

    # ------------------------------------------------------------------
    # DataFrame / query helpers
    # ------------------------------------------------------------------

    def as_dataframe(self) -> pd.DataFrame:
        """Return bars as a DataFrame with columns: timestamp, open, high, low,
        close, volume, vwap."""
        rows = [
            {
                "timestamp": b.timestamp,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": b.volume,
                "vwap": float(b.vwap) if b.vwap is not None else None,
            }
            for b in self._bars
        ]
        return pd.DataFrame(rows)

    def bars_today(self) -> list[BarEvent]:
        """Return bars whose date (ET) equals today (ET)."""
        today = datetime.now(_ET).date()
        return [
            b for b in self._bars
            if _to_et(b.timestamp).date() == today
        ]

    def since(self, dt: datetime) -> list[BarEvent]:
        """Return bars with timestamp >= *dt*."""
        return [b for b in self._bars if b.timestamp >= dt]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_et(dt: datetime) -> datetime:
    """Convert *dt* to ET, attaching the timezone if naive."""
    if dt.tzinfo is None:
        return _ET.localize(dt)
    return dt.astimezone(_ET)


# ---------------------------------------------------------------------------
# Technical indicators — standalone functions
# ---------------------------------------------------------------------------

def sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple moving average over *period* bars.

    Returns a Series of NaN where insufficient data; never raises.
    """
    try:
        return prices.rolling(window=period, min_periods=period).mean()
    except Exception:
        logger.exception("sma: unexpected error")
        return pd.Series(dtype=float)


def ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential moving average over *period* bars.

    Returns a Series of NaN where insufficient data; never raises.
    """
    try:
        return prices.ewm(span=period, adjust=False).mean()
    except Exception:
        logger.exception("ema: unexpected error")
        return pd.Series(dtype=float)


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential smoothing (alpha = 1/period).

    Returns a Series of NaN where insufficient data; never raises.
    """
    try:
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        logger.exception("rsi: unexpected error")
        return pd.Series(dtype=float)


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: (upper, middle, lower).

    middle = SMA(period); upper/lower = middle ± std_dev * rolling_std.
    Returns three Series of NaN where insufficient data; never raises.
    """
    try:
        middle = sma(prices, period)
        rolling_std = prices.rolling(window=period, min_periods=period).std()
        band = std_dev * rolling_std
        upper = middle + band
        lower = middle - band
        return upper, middle, lower
    except Exception:
        logger.exception("bollinger_bands: unexpected error")
        empty = pd.Series(dtype=float)
        return empty, empty, empty


# ---------------------------------------------------------------------------
# Market-structure / volume helpers
# ---------------------------------------------------------------------------

def vwap(history: BarHistory) -> float:
    """Intraday VWAP using only today's bars.

    Falls back to history[-1].close if no bars have volume.
    Returns NaN if history is empty.
    """
    try:
        today_bars = history.bars_today()
        if not today_bars:
            return float("nan")
        total_pv = sum(float(b.close) * b.volume for b in today_bars)
        total_vol = sum(b.volume for b in today_bars)
        if total_vol == 0:
            return float(history[-1].close)
        return total_pv / total_vol
    except Exception:
        logger.exception("vwap: unexpected error")
        return float("nan")


def anchored_vwap(history: BarHistory, anchor_index: int) -> float | None:
    """VWAP anchored to history[anchor_index].

    *anchor_index* supports negative values (e.g. -20 = 20 bars ago).
    Returns None if the index is out of range or there is no volume.
    """
    try:
        n = len(history)
        if n == 0:
            return None
        # Normalise to positive index
        if anchor_index < 0:
            pos = n + anchor_index
        else:
            pos = anchor_index
        if pos < 0 or pos >= n:
            return None
        bars = [history[i] for i in range(pos, n)]
        total_vol = sum(b.volume for b in bars)
        if total_vol == 0:
            return None
        total_pv = sum(float(b.close) * b.volume for b in bars)
        return total_pv / total_vol
    except Exception:
        logger.exception("anchored_vwap: unexpected error")
        return None


def volume_profile(
    history: BarHistory,
    num_bins: int = 20,
) -> dict[str, float] | None:
    """Coarse volume-at-price profile.

    Returns a dict with keys: poc, vah, val, hvn_levels, lvn_levels.
    Returns None if fewer than *num_bins* bars are available.
    """
    try:
        if len(history) < num_bins:
            return None

        highs = history.highs
        lows = history.lows
        closes = history.closes
        volumes = history.volumes

        price_min = lows.min()
        price_max = highs.max()
        if price_max == price_min:
            return None

        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        # Mid-price of each bucket
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bucket_vol = np.zeros(num_bins, dtype=float)

        for close, vol in zip(closes, volumes):
            idx = int((close - price_min) / bin_width)
            idx = min(idx, num_bins - 1)
            bucket_vol[idx] += vol

        poc_idx = int(np.argmax(bucket_vol))
        poc = float(bin_mids[poc_idx])

        # Value area: accumulate 70% of total volume starting from POC
        total_vol = bucket_vol.sum()
        target = 0.70 * total_vol
        order = np.argsort(bucket_vol)[::-1]  # indices sorted by vol descending
        accumulated = 0.0
        va_indices: set[int] = set()
        for idx in order:
            accumulated += bucket_vol[idx]
            va_indices.add(int(idx))
            if accumulated >= target:
                break

        va_sorted = sorted(va_indices)
        vah = float(bin_edges[va_sorted[-1] + 1])  # upper edge of highest VA bucket
        val = float(bin_edges[va_sorted[0]])        # lower edge of lowest VA bucket

        # HVN / LVN
        sorted_by_vol = list(np.argsort(bucket_vol)[::-1])
        hvn_levels = [float(bin_mids[i]) for i in sorted_by_vol[:3]]
        lvn_levels = [float(bin_mids[i]) for i in sorted_by_vol[-3:]]

        return {
            "poc": poc,
            "vah": vah,
            "val": val,
            "hvn_levels": hvn_levels,
            "lvn_levels": lvn_levels,
        }
    except Exception:
        logger.exception("volume_profile: unexpected error")
        return None


def opening_range(
    history: BarHistory,
    range_minutes: int = 30,
    market_open: str = "09:30",
) -> tuple[float, float] | tuple[None, None]:
    """High and low of the opening-range window.

    Considers only today's bars between *market_open* and
    *market_open* + *range_minutes* minutes.
    Returns (None, None) if fewer than 2 bars fall in that window.
    """
    try:
        today_et = datetime.now(_ET).date()
        open_h, open_m = (int(x) for x in market_open.split(":"))
        window_start = _ET.localize(
            datetime(today_et.year, today_et.month, today_et.day, open_h, open_m)
        )
        window_end = window_start + timedelta(minutes=range_minutes)

        bars = [
            b for b in history.bars_today()
            if window_start <= _to_et(b.timestamp) < window_end
        ]
        if len(bars) < 2:
            return None, None

        range_high = max(float(b.high) for b in bars)
        range_low = min(float(b.low) for b in bars)
        return range_high, range_low
    except Exception:
        logger.exception("opening_range: unexpected error")
        return None, None


def n_day_high(history: BarHistory, n: int = 20) -> float | None:
    """Maximum of highs over the last *n* bars.

    Returns None if len(history) < n.
    """
    try:
        if len(history) < n:
            return None
        return float(history.highs[-n:].max())
    except Exception:
        logger.exception("n_day_high: unexpected error")
        return None


def n_day_low(history: BarHistory, n: int = 20) -> float | None:
    """Minimum of lows over the last *n* bars.

    Returns None if len(history) < n.
    """
    try:
        if len(history) < n:
            return None
        return float(history.lows[-n:].min())
    except Exception:
        logger.exception("n_day_low: unexpected error")
        return None


def average_volume(history: BarHistory, period: int = 20) -> float | None:
    """Mean of volumes over the last *period* bars.

    Returns None if len(history) < period.
    """
    try:
        if len(history) < period:
            return None
        return float(history.volumes[-period:].mean())
    except Exception:
        logger.exception("average_volume: unexpected error")
        return None


# ---------------------------------------------------------------------------
# Time / session helpers
# ---------------------------------------------------------------------------

def is_market_hours(dt: datetime | None = None) -> bool:
    """Return True if *dt* (default: now ET) is during regular trading hours.

    Regular hours: Monday–Friday, 09:30–16:00 ET.
    """
    try:
        if dt is None:
            dt = datetime.now(_ET)
        dt_et = _to_et(dt)
        if dt_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= dt_et < market_close
    except Exception:
        logger.exception("is_market_hours: unexpected error")
        return False


def minutes_to_close(dt: datetime | None = None) -> float:
    """Minutes until 16:00 ET from *dt* (default: now ET).

    Returns 0.0 if the market is closed or past close.
    """
    try:
        if dt is None:
            dt = datetime.now(_ET)
        dt_et = _to_et(dt)
        close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = (close - dt_et).total_seconds() / 60.0
        return max(0.0, delta)
    except Exception:
        logger.exception("minutes_to_close: unexpected error")
        return 0.0


def time_to_flat(flat_by: str, dt: datetime | None = None) -> float:
    """Minutes until *flat_by* time ET from *dt* (default: now ET).

    *flat_by* is a "HH:MM" string, e.g. "15:45".
    Returns 0.0 if already past that time today.
    """
    try:
        if dt is None:
            dt = datetime.now(_ET)
        dt_et = _to_et(dt)
        h, m = (int(x) for x in flat_by.split(":"))
        target = dt_et.replace(hour=h, minute=m, second=0, microsecond=0)
        delta = (target - dt_et).total_seconds() / 60.0
        return max(0.0, delta)
    except Exception:
        logger.exception("time_to_flat: unexpected error")
        return 0.0
