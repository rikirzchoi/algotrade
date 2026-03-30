"""
strategies.opening_range_breakout — Crabel opening-range breakout strategy.

The opening range (OR) is defined as the high and low of the first
range_minutes minutes of the regular session (default 09:30–10:00 ET).

Phase 1 (09:30 → 09:30 + range_minutes)
  Accumulate bars; update OR high/low via core.data.opening_range().

Phase 2 (after range is set)
  Go long on the first close above OR high.
  Shorts are not implemented — long-only per Crabel convention.
  One entry per symbol per session.

Exit logic
----------
Stop     : opposite end of the OR (range_low)
Target   : entry × (1 + profit_target_pct)
Time stop: flatten all positions by flat_by time (default 15:45 ET)

All parameters are read from config.py at instantiation time.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from datetime import timedelta
from typing import Optional

import pytz

from config import AppConfig
from core.data import is_market_hours, time_to_flat
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = pytz.timezone("America/New_York")


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Crabel opening-range breakout, long-only.

    Parameters (from config.opening_range_breakout)
    ------------------------------------------------
    range_minutes     : minutes after market open that define the OR
    profit_target_pct : take-profit as a fraction of entry price
    stop_loss_pct     : fallback stop; primary stop is OR low
    flat_by           : "HH:MM" ET — flatten all positions at this time
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            strategy_id="opening_range_breakout",
            symbols=list(config.instruments.orb_symbols),
            config=config,
        )
        self._params = config.opening_range_breakout
        self.is_active = self._params.active
        self._pending_levels: dict[str, dict] = {}
        self._traded_today: dict[str, bool] = {}
        self._range_high: dict[str, float | None] = {}
        self._range_low: dict[str, float | None] = {}
        self._first_bar_logged: bool = False

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def bar_size(self) -> str:
        """Bar granularity; typically '1 min' for fine-grained ORB."""
        return self._params.bar_size

    # ------------------------------------------------------------------
    # Core callbacks
    # ------------------------------------------------------------------

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Build the opening range during Phase 1; watch for breakouts in Phase 2.

        Flattening at flat_by takes highest priority.
        Never raises — all exceptions are caught and logged internally.
        """
        try:
            # 1. Always append bar to history — no gate.
            history = self.get_history(bar.symbol)
            history.append(bar)

            # 2. Convert bar timestamp to ET once for all time comparisons.
            if bar.timestamp.tzinfo is None:
                bar_et = _ET.localize(bar.timestamp)
            else:
                bar_et = bar.timestamp.astimezone(_ET)
            bar_time = bar_et.time()
            if not self._first_bar_logged:
                logger.debug("First bar_time (ET): %s", bar_time.strftime("%H:%M"))
                self._first_bar_logged = True

            # 3. Flatten check — always, even if strategy is inactive.
            #    We must always be able to exit existing positions.
            if self.has_position(bar.symbol):
                mins = time_to_flat(self._params.flat_by, bar_et)
                if mins <= 0:
                    return self.flatten_position(bar.symbol)
                return None

            # 4. Phase 1 — range building; always runs regardless of is_active.
            phase1_start = dtime(9, 30)
            phase1_end = dtime(
                9 + (30 + self._params.range_minutes) // 60,
                (30 + self._params.range_minutes) % 60,
            )
            is_phase1 = phase1_start <= bar_time < phase1_end

            if is_phase1:
                session_start = bar_et.replace(hour=9, minute=30, second=0, microsecond=0)
                phase1_bars = history.since(session_start)
                if phase1_bars:
                    self._range_high[bar.symbol] = max(float(b.high) for b in phase1_bars)
                    self._range_low[bar.symbol] = min(float(b.low) for b in phase1_bars)
                return None  # never trade during phase 1

            # 5. Gate order generation on is_active.
            if not self._params.active:
                return None

            # 6. Phase 2 — breakout detection.
            if not is_market_hours(bar_et):
                return None
            if self._traded_today.get(bar.symbol, False):
                return None
            if self.has_position(bar.symbol):
                return None
            if time_to_flat(self._params.flat_by, bar_et) <= 15:
                return None

            range_high = self._range_high.get(bar.symbol)
            range_low = self._range_low.get(bar.symbol)
            if range_high is None or range_low is None:
                return None

            close = float(bar.close)

            # LONG entry: close breaks above OR high.
            if close > range_high:
                profit_target = close * (1 + self._params.profit_target_pct)
                stop_loss = range_low  # stop at the opposite end of the range
                reason = (
                    f"ORB long: close={close:.2f} > "
                    f"range_high={range_high:.2f}, "
                    f"range={range_high - range_low:.2f}"
                )
                self._traded_today[bar.symbol] = True
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
                    strength=0.8,
                    reason=reason,
                )

            # SHORT entry: not implemented (long-only).
            return None
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
                self._range_high.pop(fill.symbol, None)
                self._range_low.pop(fill.symbol, None)
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

    def reset_daily(self) -> None:
        """Clear per-day session state.  Called by the engine at market open."""
        self._traded_today = {}
        self._range_high = {}
        self._range_low = {}
