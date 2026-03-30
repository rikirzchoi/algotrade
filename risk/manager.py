"""
risk.manager — real-time risk enforcement layer.

The RiskManager sits between the engine's signal handler and the broker.
Every SignalEvent must pass through approve_signal() before an OrderEvent is
created.  The manager enforces:

  - Kill switch: manual or automatic; blocks all new entry orders when active
  - Duplicate-position filter: suppresses a new entry if a position is already
    open in that symbol across all strategies combined
  - Daily loss limit: once breached, new entries are blocked and the kill
    switch is automatically triggered
  - Maximum drawdown: once exceeded, the kill switch fires
  - Fixed-fractional position sizing with optional macro regime multiplier

All shared state is protected by threading.Lock.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

from config import AppConfig
from core.events import Direction, FillEvent, SignalEvent

logger = logging.getLogger("risk.manager")


class RiskManager:
    """
    Stateful risk gate between strategy signals and order execution.

    Parameters come from AppConfig.risk; the engine passes AppConfig at
    construction time.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._peak_equity: float = config.risk.capital
        self._current_equity: float = config.risk.capital
        self._daily_pnl: float = 0.0
        self._is_halted: bool = False
        self._logger = logging.getLogger("risk.manager")

    # ------------------------------------------------------------------
    # Public API called by the engine
    # ------------------------------------------------------------------

    def approve_signal(
        self,
        signal: SignalEvent,
        all_positions: dict[str, int],
    ) -> tuple[bool, str]:
        """
        Validate *signal* against all risk rules.

        Returns (True, "approved") if the signal passes all checks, or
        (False, reason) where reason describes the rejection.

        Checks are evaluated in this order:
          1. Kill switch active
          2. FLAT signals always approved
          3. Symbol already has an open position
          4. Daily loss limit reached
          5. Max drawdown exceeded
        """
        with self._lock:
            if self._is_halted:
                return False, "kill switch active — engine is halted"

            if signal.direction == Direction.FLAT:
                return True, "approved"

            if all_positions.get(signal.symbol, 0) != 0:
                return False, f"already in position for {signal.symbol}"

            if self._daily_pnl <= -self._config.risk.max_daily_loss_usd:
                return (
                    False,
                    f"daily loss limit reached: {self._daily_pnl:.2f}",
                )

            drawdown = self._current_drawdown_pct_unlocked()
            if drawdown > self._config.risk.max_drawdown_pct:
                return (
                    False,
                    f"drawdown limit reached: {drawdown:.1%}",
                )

            return True, "approved"

    def size_position(
        self,
        signal: SignalEvent,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """
        Calculate position size using fixed-fractional risk sizing.

        Optionally applies a macro regime multiplier if the regime module is
        available; silently skips it if the import fails.
        """
        capital = self._config.risk.capital
        risk_per_trade_pct = self._config.risk.risk_per_trade_pct
        max_position_size = self._config.risk.max_position_size

        risk_dollars = capital * risk_per_trade_pct
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share <= 0:
            return 1

        qty = math.floor(risk_dollars / risk_per_share)
        qty = max(1, min(qty, max_position_size))

        try:
            from macro.regime import get_position_size_multiplier  # type: ignore
            multiplier = get_position_size_multiplier()
            qty = math.floor(qty * multiplier)
        except Exception:
            pass

        return qty

    def record_fill(self, fill: FillEvent) -> None:
        """
        Called on entry fills — log the fill; full PnL tracking on close via
        record_close().
        """
        with self._lock:
            self._logger.info(
                "Fill recorded: strategy=%s symbol=%s direction=%s qty=%d "
                "price=%s commission=%s exec_id=%s",
                fill.strategy_id,
                fill.symbol,
                fill.direction.value,
                fill.quantity,
                fill.fill_price,
                fill.commission,
                fill.ib_exec_id,
            )

    def record_close(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: Direction,
    ) -> None:
        """
        Calculate trade PnL and update equity/drawdown state.

        Triggers the kill switch automatically if daily loss or drawdown
        thresholds are breached.
        """
        if direction == Direction.LONG:
            pnl = (exit_price - entry_price) * quantity
        elif direction == Direction.SHORT:
            pnl = (entry_price - exit_price) * quantity
        else:
            pnl = 0.0

        with self._lock:
            self._daily_pnl += pnl
            self._current_equity += pnl
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity

            daily_pnl_snap = self._daily_pnl
            drawdown_snap = self._current_drawdown_pct_unlocked()

        if daily_pnl_snap <= -self._config.risk.max_daily_loss_usd:
            self.trigger_halt("daily loss limit")
        elif drawdown_snap >= self._config.risk.max_drawdown_pct:
            self.trigger_halt("max drawdown")

    def trigger_halt(self, reason: str) -> None:
        """Activate the kill switch and log a CRITICAL alert."""
        with self._lock:
            self._is_halted = True
            self._logger.critical(
                "KILL SWITCH TRIGGERED: %s | daily_pnl=%.2f drawdown=%.1f%%",
                reason,
                self._daily_pnl,
                self._current_drawdown_pct_unlocked() * 100,
            )

    def reset_daily(self) -> None:
        """Reset daily PnL at the start of a new session. Equity is preserved."""
        with self._lock:
            self._daily_pnl = 0.0
            self._logger.info(
                "Daily PnL reset. Peak equity: %.2f", self._peak_equity
            )

    def resume(self) -> None:
        """Manually clear the kill switch. Use with caution."""
        with self._lock:
            self._is_halted = False
            self._logger.warning("Kill switch manually cleared. Monitor closely.")

    # ------------------------------------------------------------------
    # Properties (all thread-safe)
    # ------------------------------------------------------------------

    @property
    def daily_pnl(self) -> float:
        with self._lock:
            return self._daily_pnl

    @property
    def peak_equity(self) -> float:
        with self._lock:
            return self._peak_equity

    @property
    def current_equity(self) -> float:
        with self._lock:
            return self._current_equity

    @property
    def current_drawdown_pct(self) -> float:
        """Fraction of peak equity lost. Returns 0.0 if peak_equity is zero."""
        with self._lock:
            return self._current_drawdown_pct_unlocked()

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._is_halted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_drawdown_pct_unlocked(self) -> float:
        """Compute drawdown without acquiring the lock (caller must hold it)."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity
