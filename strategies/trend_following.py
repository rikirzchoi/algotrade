"""
strategies.trend_following — EMA crossover trend following for commodity ETFs.

Entry logic
-----------
LONG  when the fast EMA crosses above the slow EMA on the current bar
      (previous bar: fast ≤ slow; current bar: fast > slow).
SHORT when the fast EMA crosses below the slow EMA on the current bar.

Stop / target (both ATR-based)
-------------------------------
LONG  stop  : entry − atr_stop_multiplier   × ATR(atr_period)
LONG  target: entry + atr_target_multiplier × ATR(atr_period)
SHORT stop  : entry + atr_stop_multiplier   × ATR(atr_period)
SHORT target: entry − atr_target_multiplier × ATR(atr_period)

Rationale: commodity ETFs (GLD, USO) move in cleaner macro-driven waves
than equity indices, making trend following on the 4-hour chart more
reliable than mean-reversion approaches.  Using ATR for stop/target
means position size (computed by fixed-fractional) automatically shrinks
in quiet markets and grows in trending ones.

All parameters are read from config.py at instantiation time.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from config import AppConfig
from core.data import atr, ema
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """
    EMA crossover trend-following strategy for commodity ETFs.

    Parameters (from config.trend_following)
    -----------------------------------------
    fast_ema              : fast EMA period (default 21)
    slow_ema              : slow EMA period (default 55)
    atr_period            : ATR look-back for stop/target sizing (default 14)
    atr_stop_multiplier   : stop distance = multiplier × ATR
    atr_target_multiplier : target distance = multiplier × ATR
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            strategy_id="trend_following",
            symbols=list(config.instruments.commodity_symbols),
            config=config,
        )
        self._params = config.trend_following
        self.is_active = self._params.active
        self._pending_levels: dict[str, dict] = {}
        # Tracks whether fast EMA was above slow EMA on the previous bar
        # to detect the crossover moment.  None = no prior bar seen.
        self._prev_fast_above: dict[str, bool | None] = {}

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def bar_size(self) -> str:
        """Bar granularity; '4 hours' for this strategy."""
        return self._params.bar_size

    # ------------------------------------------------------------------
    # Core callbacks
    # ------------------------------------------------------------------

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Detect EMA crossover; emit LONG or SHORT SignalEvent on the crossing bar.

        Never raises — all exceptions are caught and logged internally.
        """
        try:
            if not self._params.active:
                return None

            if self.has_position(bar.symbol):
                return None

            history = self.get_history(bar.symbol)
            history.append(bar)

            params = self._params
            # Need enough bars to compute the slow EMA and one full ATR window.
            min_bars = params.slow_ema + params.atr_period + 1
            if len(history) < min_bars:
                return None

            closes = pd.Series(history.closes)
            highs = pd.Series(history.highs)
            lows = pd.Series(history.lows)

            fast_series = ema(closes, params.fast_ema)
            slow_series = ema(closes, params.slow_ema)

            if (
                fast_series.empty
                or slow_series.empty
                or pd.isna(fast_series.iloc[-1])
                or pd.isna(slow_series.iloc[-1])
            ):
                return None

            atr_val = atr(highs, lows, closes, params.atr_period)
            if math.isnan(atr_val) or atr_val <= 0:
                return None

            fast_now = float(fast_series.iloc[-1])
            slow_now = float(slow_series.iloc[-1])
            fast_above_now = fast_now > slow_now

            symbol = bar.symbol
            prev_fast_above = self._prev_fast_above.get(symbol)
            self._prev_fast_above[symbol] = fast_above_now

            # Crossovers only fire on the bar where the relationship changes.
            if prev_fast_above is None:
                return None

            close = float(bar.close)

            if fast_above_now and not prev_fast_above:
                # Bullish cross → LONG
                stop = close - atr_val * params.atr_stop_multiplier
                target = close + atr_val * params.atr_target_multiplier
                direction = Direction.LONG
                reason = (
                    f"EMA cross LONG: fast={fast_now:.3f} > slow={slow_now:.3f}, "
                    f"ATR={atr_val:.3f}"
                )
            elif not fast_above_now and prev_fast_above:
                # Bearish cross → SHORT
                stop = close + atr_val * params.atr_stop_multiplier
                target = close - atr_val * params.atr_target_multiplier
                direction = Direction.SHORT
                reason = (
                    f"EMA cross SHORT: fast={fast_now:.3f} < slow={slow_now:.3f}, "
                    f"ATR={atr_val:.3f}"
                )
            else:
                return None

            self._pending_levels[symbol] = {
                "entry": close,
                "stop": stop,
                "target": target,
            }

            # Signal strength: how far apart are the EMAs relative to ATR?
            separation = abs(fast_now - slow_now) / atr_val
            strength = min(1.0, separation)

            return SignalEvent(
                strategy_id=self.strategy_id,
                symbol=symbol,
                timestamp=bar.timestamp,
                direction=direction,
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
