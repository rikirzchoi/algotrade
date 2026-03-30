"""
strategies.momentum_breakout — N-day high breakout with volume and anchored VWAP confirmation.

Entry logic
-----------
Go long when:
  - The current close exceeds the highest high of the previous N bars
  - Bar volume is at least volume_multiplier × N-bar average volume
  - Close is above the anchored VWAP from N bars ago

One trade per symbol per session.  Stop / target levels are stored in
_pending_levels so the engine can retrieve them for bracket order placement.

All parameters are read from config.py at instantiation time.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import AppConfig
from core.data import anchored_vwap, average_volume
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumBreakoutStrategy(BaseStrategy):
    """
    N-day high breakout strategy with volume and anchored VWAP confirmation.

    Parameters (from config.momentum_breakout)
    -------------------------------------------
    lookback_days      : N defining the "N-day high" and lookback window
    volume_multiplier  : bar volume must exceed avg_vol × this to confirm
    profit_target_pct  : take-profit distance as a fraction of entry price
    stop_loss_pct      : stop distance as a fraction of entry price
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            strategy_id="momentum_breakout",
            symbols=list(config.instruments.swing_symbols),
            config=config,
        )
        self._params = config.momentum_breakout
        self.is_active = self._params.active
        self._traded_today: dict[str, bool] = {}
        self._pending_levels: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def bar_size(self) -> str:
        """Bar granularity; typically '1 day' for swing strategies."""
        return self._params.bar_size

    # ------------------------------------------------------------------
    # Core callbacks
    # ------------------------------------------------------------------

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Evaluate N-day high breakout conditions; return SignalEvent or None.

        Never raises — all exceptions are caught and logged internally.
        """
        try:
            if not self._params.active:
                return None

            # Daily bars arrive after market close; skip the live-hours gate.

            if self._traded_today.get(bar.symbol, False):
                return None

            if self.has_position(bar.symbol):
                return None

            history = self.get_history(bar.symbol)
            history.append(bar)

            params = self._params
            # Need lookback_days historical bars plus the current one.
            if len(history) < params.lookback_days + 1:
                return None

            # N-day high from bars BEFORE the current bar so we can test
            # whether today's close has genuinely broken out.
            nd_high = float(history.highs[:-1][-params.lookback_days:].max())
            avg_vol = average_volume(history, params.lookback_days)
            avwap = anchored_vwap(history, anchor_index=-params.lookback_days)

            if avg_vol is None or avwap is None:
                return None

            close = float(bar.close)
            vol = bar.volume

            if not (
                close > nd_high
                and vol > avg_vol * params.volume_multiplier
                and close > avwap
            ):
                return None

            profit_target = close * (1 + params.profit_target_pct)
            stop_loss = close * (1 - params.stop_loss_pct)
            strength = min(1.0, vol / (avg_vol * params.volume_multiplier))
            reason = (
                f"20d high breakout: close={close:.2f} > "
                f"high={nd_high:.2f}, vol={vol / avg_vol:.1f}x avg"
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

    def reset_daily(self) -> None:
        """Clear per-day traded flags.  Called by the engine at market open."""
        self._traded_today = {}

    def get_pending_levels(self, symbol: str) -> dict | None:
        """Return {'entry', 'stop', 'target'} for *symbol*, or None.

        The engine calls this after receiving a signal to build the bracket order.
        """
        return self._pending_levels.get(symbol)
