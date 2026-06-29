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
        self._halt_reason: Optional[str] = None
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
          4. Correlation filter (too many correlated risk-on positions already open)
          5. Daily loss limit reached
          6. Max drawdown exceeded
        """
        with self._lock:
            if self._is_halted:
                return False, "kill switch active — engine is halted"

            if signal.direction == Direction.FLAT:
                return True, "approved"

            if all_positions.get(signal.symbol, 0) != 0:
                return False, f"already in position for {signal.symbol}"

            if signal.direction == Direction.LONG:
                risk_on = self._config.risk.risk_on_symbols
                limit = self._config.risk.max_risk_on_positions
                if signal.symbol in risk_on:
                    open_risk_on = sum(
                        1 for s in risk_on if all_positions.get(s, 0) > 0
                    )
                    if open_risk_on >= limit:
                        return (
                            False,
                            f"correlation filter: {open_risk_on} risk-on positions "
                            f"already open (limit {limit})",
                        )

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
        from risk.sizing import fixed_fractional

        # Core fixed-fractional math — the single shared sizing function also
        # used by the backtester, so backtest and live size identically.
        qty = fixed_fractional(
            self._config.risk.capital,
            self._config.risk.risk_per_trade_pct,
            entry_price,
            stop_loss_price,
            self._config.risk.max_position_size,
        )

        # Macro regime overlay (live + backtest both apply it via this path).
        try:
            from macro.regime import get_position_size_multiplier, load_regime
            multiplier = get_position_size_multiplier(load_regime())
            qty = math.floor(qty * multiplier)
        except Exception:
            self._logger.exception(
                "macro multiplier failed; using unscaled size"
            )

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
        commission: float = 0.0,
    ) -> float:
        """
        Calculate trade PnL and update equity/drawdown state.

        Triggers the kill switch automatically if daily loss or drawdown
        thresholds are breached.

        Parameters
        ----------
        direction : the side of the *position* being closed (LONG or SHORT),
                    not the side of the closing execution.
        commission : total round-trip commission to deduct from realised PnL.

        Returns
        -------
        float
            Realised net PnL for this close (after commission).
        """
        if direction == Direction.LONG:
            pnl = (exit_price - entry_price) * quantity
        elif direction == Direction.SHORT:
            pnl = (entry_price - exit_price) * quantity
        else:
            pnl = 0.0

        pnl -= commission

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

        return pnl

    def trigger_halt(self, reason: str) -> None:
        """Activate the kill switch and log a CRITICAL alert."""
        with self._lock:
            self._is_halted = True
            self._halt_reason = reason
            self._logger.critical(
                "KILL SWITCH TRIGGERED: %s | daily_pnl=%.2f drawdown=%.1f%%",
                reason,
                self._daily_pnl,
                self._current_drawdown_pct_unlocked() * 100,
            )

    def reset_daily(self) -> None:
        """Reset daily PnL at the start of a new session. Equity is preserved.

        A **daily-loss** halt is a daily circuit breaker — it clears on the new
        day (you'd resume next morning). A **drawdown** halt is persistent and
        only a manual resume() clears it.
        """
        with self._lock:
            self._daily_pnl = 0.0
            if self._is_halted and self._halt_reason == "daily loss limit":
                self._is_halted = False
                self._halt_reason = None
                self._logger.info("New day — daily-loss halt cleared.")
            self._logger.info(
                "Daily PnL reset. Peak equity: %.2f", self._peak_equity
            )

    def resume(self) -> None:
        """Manually clear the kill switch. Use with caution."""
        with self._lock:
            self._is_halted = False
            self._halt_reason = None
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
