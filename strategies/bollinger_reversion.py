"""
strategies.bollinger_reversion — Bollinger Band + RSI mean-reversion strategy.

Entry logic (band touch + recovery)
-------------------------------------
Go long when ALL of the following are true:
  - Previous bar closed at or below the lower Bollinger Band (band touch)
  - Current bar closes above the lower band (recovery back inside bands)
  - RSI on current bar is below rsi_oversold + 10 (relaxed threshold)
  - Close falls within the volume-profile Value Area (val ≤ close ≤ vah)

Exit logic
----------
Profit target : middle band (20-period SMA) at signal time
Hard stop     : below the lowest low of the two most recent bars × 0.999
Time stop     : flatten all positions by flat_by time (default 15:45 ET)

All parameters are read from config.py at instantiation time.
"""

from __future__ import annotations

import logging
from datetime import date
from statistics import mean
from typing import Optional

import pandas as pd
import pytz

from config import AppConfig
from core.data import bollinger_bands, is_market_hours, rsi, time_to_flat, volume_profile
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")


class BollingerReversionStrategy(BaseStrategy):
    """
    Bollinger Band + RSI mean-reversion with volume-profile Value Area filter.

    Parameters (from config.bollinger_reversion)
    ---------------------------------------------
    bb_period       : Bollinger Band SMA period
    bb_std          : standard-deviation multiplier for the bands
    rsi_period      : RSI look-back period
    rsi_oversold    : RSI threshold below which the instrument is oversold
    profit_target_pct : fallback target; primary target is the middle band
    stop_loss_pct   : stop distance as a fraction of entry price
    flat_by         : "HH:MM" ET — flatten all positions at this time
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            strategy_id="bollinger_reversion",
            symbols=list(config.instruments.bollinger_symbols),
            config=config,
        )
        self._params = config.bollinger_reversion
        self.is_active = self._params.active
        self._pending_levels: dict[str, dict] = {}
        self._prev_lower: dict[str, float] = {}
        self._daily_trend: dict[str, str] = {}
        self._daily_trend_date: dict[str, date] = {}
        self._daily_closes: dict[str, list[float]] = {}
        self._last_stop_date: dict[str, date] = {}

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def bar_size(self) -> str:
        """Bar granularity; typically '15 mins' for this intraday strategy."""
        return self._params.bar_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_daily_trend(self, bar: BarEvent) -> None:
        """Maintain a daily trend state (up/down/neutral) for *bar.symbol*.

        Called once per bar.  Updates only when the calendar date (ET) changes,
        so it runs at most once per trading day regardless of bar frequency.
        """
        if bar.timestamp.tzinfo is None:
            bar_et = _ET.localize(bar.timestamp)
        else:
            bar_et = bar.timestamp.astimezone(_ET)

        today = bar_et.date()
        symbol = bar.symbol

        if self._daily_trend_date.get(symbol) == today:
            return  # already updated for today

        # New trading day: record yesterday's close and recompute trend.
        closes = self._daily_closes.setdefault(symbol, [])
        closes.append(float(bar.close))

        if len(closes) >= 10:
            sma_10 = mean(closes[-10:])
            sma_5 = mean(closes[-5:])
            if sma_5 > sma_10:
                self._daily_trend[symbol] = "up"
            elif sma_5 < sma_10:
                self._daily_trend[symbol] = "down"
            else:
                self._daily_trend[symbol] = "neutral"
        else:
            self._daily_trend[symbol] = "neutral"

        self._daily_trend_date[symbol] = today

    # ------------------------------------------------------------------
    # Core callbacks
    # ------------------------------------------------------------------

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Flatten at flat_by time; otherwise evaluate entry conditions.

        Never raises — all exceptions are caught and logged internally.
        """
        try:
            self._update_daily_trend(bar)

            # Convert bar timestamp to ET once for all time comparisons.
            if bar.timestamp.tzinfo is None:
                bar_et = _ET.localize(bar.timestamp)
            else:
                bar_et = bar.timestamp.astimezone(_ET)

            # Flatten check — highest priority; checked before entry logic.
            if self.has_position(bar.symbol):
                mins = time_to_flat(self._params.flat_by, bar_et)
                if mins <= 0:
                    return self.flatten_position(bar.symbol)
                # Managing an open position; no new signals.
                return None

            if not self._params.active:
                return None
            if not is_market_hours(bar_et):
                return None

            # Same-day re-entry block: skip if already stopped out today.
            if self._last_stop_date.get(bar.symbol) == bar_et.date():
                return None
            # Don't open new positions within 15 minutes of the flat time.
            if time_to_flat(self._params.flat_by, bar_et) <= 15:
                return None

            # Time filter: only allow entries in the two permitted windows.
            bar_time = bar_et.time()
            in_morning = (
                bar_time >= bar_et.replace(hour=9, minute=30).time()
                and bar_time < bar_et.replace(hour=11, minute=30).time()
            )
            in_afternoon = (
                bar_time >= bar_et.replace(hour=13, minute=30).time()
                and bar_time < bar_et.replace(hour=15, minute=15).time()
            )
            entry_window = in_morning or in_afternoon

            history = self.get_history(bar.symbol)
            history.append(bar)

            params = self._params
            min_bars = params.bb_period + params.rsi_period
            if len(history) < min_bars:
                return None

            closes = pd.Series(history.closes)
            upper, middle, lower = bollinger_bands(closes, params.bb_period, params.bb_std)
            rsi_vals = rsi(closes, params.rsi_period)
            vp = volume_profile(history, num_bins=20)

            # Guard against NaN from insufficient data.
            if lower.empty or pd.isna(lower.iloc[-1]):
                self._prev_lower[bar.symbol] = float("nan")
                return None
            if rsi_vals.empty or pd.isna(rsi_vals.iloc[-1]):
                self._prev_lower[bar.symbol] = float(lower.iloc[-1])
                return None

            close = float(bar.close)
            lower_val = float(lower.iloc[-1])
            upper_val = float(upper.iloc[-1])
            middle_val = float(middle.iloc[-1])
            rsi_val = float(rsi_vals.iloc[-1])
            prev_bar = history[-2]
            prev_lower_val = self._prev_lower.get(bar.symbol)

            # Store current lower band for the next bar before any early return.
            self._prev_lower[bar.symbol] = lower_val

            if not entry_window:
                return None

            # Daily trend filter: never buy into a sustained downtrend.
            daily_trend = self._daily_trend.get(bar.symbol, "neutral")
            if daily_trend == "down":
                return None

            # Confirmation candle: previous bar touched the band, current bar
            # has recovered above it, RSI still in oversold zone.
            if prev_lower_val is None:
                return None
            if not (
                float(prev_bar.close) <= prev_lower_val
                and close > lower_val
                and rsi_val < params.rsi_oversold + 10.0
                and vp is not None
                and vp["val"] <= close <= vp["vah"]
            ):
                return None

            profit_target = middle_val  # mean-reversion target is the middle band
            stop_loss = min(float(prev_bar.low), float(bar.low)) * 0.999
            band_width = upper_val - lower_val
            bb_pct = (close - lower_val) / band_width if band_width != 0 else 0.0
            strength = max(0.1, 1.0 - bb_pct)
            reason = (
                f"BB touch+recovery: prev_close={float(prev_bar.close):.2f} touched "
                f"lower={prev_lower_val:.2f}, "
                f"close={close:.2f} recovering, "
                f"RSI={rsi_val:.1f}, "
                f"target={profit_target:.2f}"
            )

            self._pending_levels[bar.symbol] = {
                "entry": close,
                "stop": stop_loss,
                "target": profit_target,
            }

            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                direction=Direction.LONG,
                strength=strength,
                reason=reason,
            )
        except Exception:
            self._logger.exception("on_bar error for %s", bar.symbol)
            return None

    def on_fill(self, fill: FillEvent) -> None:
        """Update position tracking on fill confirmation."""
        try:
            if fill.direction == Direction.LONG:
                self.update_position(fill.symbol, fill.quantity)
            elif fill.direction == Direction.SHORT:
                self.update_position(fill.symbol, -fill.quantity)
            elif fill.direction == Direction.FLAT:
                self.update_position(fill.symbol, -self.get_position(fill.symbol))
                self._pending_levels.pop(fill.symbol, None)
                self._last_stop_date[fill.symbol] = fill.timestamp.astimezone(_ET).date()
            self._logger.info(
                "Fill: %s %s qty=%d @ %.2f",
                fill.direction.value,
                fill.symbol,
                fill.quantity,
                float(fill.fill_price),
            )
        except Exception:
            self._logger.exception("on_fill error for %s", fill.symbol)

    # ------------------------------------------------------------------
    # Engine-facing helpers
    # ------------------------------------------------------------------

    def get_pending_levels(self, symbol: str) -> dict | None:
        """Return {'entry', 'stop', 'target'} for *symbol*, or None.

        The engine calls this after receiving a signal to build the bracket order.
        """
        return self._pending_levels.get(symbol)
