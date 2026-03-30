"""
core.engine — central event loop that wires every component together.

The TradingEngine owns a single inbound queue (provided by the Broker).  The
main thread drains that queue and dispatches each event to the correct handler:

  ("historical", BarEvent)  → warm-up fan-out to strategies (no orders)
  ("realtime",  BarEvent)   → live fan-out; signals may produce bracket orders
  FillEvent                 → strategy + risk manager notified; state updated
  SystemEvent               → logged; ENGINE_STOP / KILL_SWITCH triggers stop()

The engine never imports ibapi directly.  All IB interaction is delegated to
core.broker.Broker.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

from zoneinfo import ZoneInfo

from config import AppConfig
from core.broker import Broker
from core.events import (
    BarEvent,
    FillEvent,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from database.writer import DatabaseWriter
from risk.manager import RiskManager

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")


class TradingEngine:
    """
    Central event dispatcher and lifecycle manager.

    Usage::

        engine = TradingEngine(cfg)
        engine.register_strategy(MomentumBreakoutStrategy(cfg))
        engine.run()   # blocks until stop() is called or ENGINE_STOP received
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._broker = Broker(config)
        self._risk = RiskManager(config)
        self._db = DatabaseWriter(
            config.database.db_path,
            config.database.flush_interval_seconds,
        )
        self._strategies: list[BaseStrategy] = []
        self._stop_event = threading.Event()
        self._logger = logging.getLogger("engine")

        # Shared state — dashboard reads via shared_state property
        self._state: dict = {
            "is_running": False,
            "is_halted": False,
            "daily_pnl": 0.0,
            "peak_equity": config.risk.capital,
            "drawdown_pct": 0.0,
            "last_heartbeat": None,
            "positions": {},
            "active_strategies": [],
            "fills_today": [],
            "error_count_today": 0,
        }
        self._state_lock = threading.Lock()

        # Daily reset tracking
        self._reset_done_today: bool = False
        self._reset_date: object = None  # datetime.date | None

        # Periodic state update tracking
        self._last_periodic_update: float = 0.0

    # ------------------------------------------------------------------
    # Strategy registration
    # ------------------------------------------------------------------

    def register_strategy(self, strategy: "BaseStrategy") -> None:
        """Append a strategy to the dispatch list and log registration."""
        self._strategies.append(strategy)
        self._logger.info(
            "Registered strategy: %s watching %s",
            strategy.strategy_id,
            strategy.symbols,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all components and enter the main event loop (blocks)."""
        self._db.start()
        self._broker.connect()

        # Request historical (+ live) data for every unique (symbol, bar_size)
        req_id = 1
        seen: set[tuple[str, str]] = set()
        for strategy in self._strategies:
            for symbol in strategy.symbols:
                key = (symbol, strategy.bar_size)
                if key in seen:
                    continue
                seen.add(key)
                duration = "1 Y" if strategy.bar_size == "1 day" else "5 D"
                self._broker.request_historical_data(
                    req_id, symbol, strategy.bar_size, duration
                )
                req_id += 1

        with self._state_lock:
            self._state["is_running"] = True

        cfg = self._config
        all_symbols = sorted({s for st in self._strategies for s in st.symbols})
        self._logger.info("=== AlgoTrade Engine Started ===")
        self._logger.info("Mode: %s", "PAPER" if cfg.is_paper_trading else "LIVE")
        self._logger.info(
            "Strategies: %s", [s.strategy_id for s in self._strategies]
        )
        self._logger.info("Symbols: %s", all_symbols)
        self._logger.info(
            "Risk limits: daily_loss=$%.2f, drawdown=%.0f%%",
            cfg.risk.max_daily_loss_usd,
            cfg.risk.max_drawdown_pct * 100,
        )

        while not self._stop_event.is_set():
            try:
                item = self._broker.event_queue.get(timeout=1.0)
            except queue.Empty:
                self._check_market_open_reset()
                self._maybe_update_state_periodic()
                continue
            self._dispatch(item)
            self._maybe_update_state_periodic()

    def stop(self) -> None:
        """Signal the loop to exit, cancel orders, and disconnect cleanly."""
        self._stop_event.set()
        self._broker.cancel_all_orders()
        self._broker.disconnect()
        self._db.stop()
        with self._state_lock:
            self._state["is_running"] = False
        self._logger.info("Engine stopped cleanly")

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def toggle_strategy(self, strategy_id: str, active: bool) -> None:
        """Enable or disable a strategy by its strategy_id."""
        for strategy in self._strategies:
            if strategy.strategy_id == strategy_id:
                strategy.is_active = active
                self._logger.info(
                    "Strategy %s %s",
                    strategy_id,
                    "activated" if active else "deactivated",
                )
                with self._state_lock:
                    self._state["active_strategies"] = [
                        s.strategy_id for s in self._strategies if s.is_active
                    ]
                return

    @property
    def shared_state(self) -> dict:
        """Return a shallow copy of internal state (thread-safe)."""
        with self._state_lock:
            return dict(self._state)

    @property
    def risk_manager(self) -> RiskManager:
        """Read-only access for the dashboard kill switch button."""
        return self._risk

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, item: object) -> None:
        """Route one queue item to the appropriate handler."""
        if isinstance(item, tuple) and len(item) == 2:
            kind, bar = item
            if not isinstance(bar, BarEvent):
                return
            if kind == "historical":
                self._db.write(bar)
                for strategy in self._strategies:
                    if bar.symbol in strategy.symbols and strategy.is_active:
                        strategy.on_bar(bar)
                        # Historical bars warm up indicators only — no orders
            elif kind == "realtime":
                self._db.write(bar)
                for strategy in self._strategies:
                    if bar.symbol in strategy.symbols and strategy.is_active:
                        signal = strategy.on_bar(bar)
                        if signal is not None:
                            self._handle_signal(signal, strategy)
        elif isinstance(item, FillEvent):
            self._db.write(item)
            for strategy in self._strategies:
                if strategy.strategy_id == item.strategy_id:
                    strategy.on_fill(item)
            self._risk.record_fill(item)
            self._update_state_on_fill(item)
        elif isinstance(item, SystemEvent):
            self._db.write(item)
            if item.kind in (
                SystemEventKind.ENGINE_STOP,
                SystemEventKind.KILL_SWITCH,
            ):
                self.stop()
            elif (
                item.kind == SystemEventKind.WARNING
                and "heartbeat" in item.message
            ):
                with self._state_lock:
                    self._state["last_heartbeat"] = item.timestamp
            elif item.kind == SystemEventKind.ERROR:
                with self._state_lock:
                    self._state["error_count_today"] += 1
                self._logger.error(item.message)

    def _handle_signal(
        self, signal: SignalEvent, strategy: "BaseStrategy"
    ) -> None:
        """Run risk checks, size position, and place a bracket order."""
        all_positions = self._aggregate_positions()
        approved, reason = self._risk.approve_signal(signal, all_positions)
        self._db.write(signal)
        if not approved:
            self._logger.info("Signal rejected: %s", reason)
            return

        levels = strategy.get_pending_levels(signal.symbol)
        if levels is None:
            self._logger.warning(
                "Signal approved but no pending levels for %s", signal.symbol
            )
            return

        qty = self._risk.size_position(signal, levels["entry"], levels["stop"])
        client_order_id = (
            f"{signal.strategy_id}_{signal.symbol}_{int(time.time())}"
        )
        self._broker.place_bracket_order(
            client_order_id=client_order_id,
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=qty,
            entry_price=levels["entry"],
            profit_target=levels["target"],
            stop_loss=levels["stop"],
        )
        self._logger.info(
            "Order placed: %s %s %d %s @ %.2f stop=%.2f target=%.2f",
            signal.strategy_id,
            signal.direction.name,
            qty,
            signal.symbol,
            levels["entry"],
            levels["stop"],
            levels["target"],
        )

    def _aggregate_positions(self) -> dict[str, int]:
        """Merge per-strategy positions into one combined dict."""
        combined: dict[str, int] = {}
        for strategy in self._strategies:
            for symbol, qty in strategy.positions.items():
                combined[symbol] = combined.get(symbol, 0) + qty
        return combined

    def _check_market_open_reset(self) -> None:
        """Call daily reset once per day inside the 09:29–09:31 ET window."""
        now = datetime.now(tz=_ET)
        today = now.date()

        # Advance flag whenever the calendar date rolls over (midnight ET)
        if self._reset_date is not None and self._reset_date != today:
            self._reset_done_today = False
        self._reset_date = today

        if self._reset_done_today:
            return

        if now.hour == 9 and 29 <= now.minute <= 31:
            self._risk.reset_daily()
            for strategy in self._strategies:
                if hasattr(strategy, "reset_daily"):
                    strategy.reset_daily()
            self._reset_done_today = True
            self._logger.info("Daily reset complete")

    def _maybe_update_state_periodic(self) -> None:
        """Refresh shared state approximately every 5 seconds."""
        now = time.monotonic()
        if now - self._last_periodic_update >= 5.0:
            self._update_state_periodic()
            self._last_periodic_update = now

    def _update_state_on_fill(self, fill: FillEvent) -> None:
        """Thread-safe state update triggered by a fill."""
        positions = self._aggregate_positions()
        with self._state_lock:
            self._state["positions"] = positions
            self._state["fills_today"].append(fill)
            if len(self._state["fills_today"]) > 100:
                self._state["fills_today"] = self._state["fills_today"][-100:]
            self._state["daily_pnl"] = self._risk.daily_pnl
            self._state["drawdown_pct"] = self._risk.current_drawdown_pct
            self._state["is_halted"] = self._risk.is_halted

    def _update_state_periodic(self) -> None:
        """Refresh dashboard-facing counters from risk manager properties."""
        with self._state_lock:
            self._state["daily_pnl"] = self._risk.daily_pnl
            self._state["peak_equity"] = self._risk.peak_equity
            self._state["drawdown_pct"] = self._risk.current_drawdown_pct
            self._state["is_halted"] = self._risk.is_halted
            self._state["active_strategies"] = [
                s.strategy_id for s in self._strategies if s.is_active
            ]


# ---------------------------------------------------------------------------
# Public API reference card
# ---------------------------------------------------------------------------
#
# TradingEngine(config: AppConfig)
#
# register_strategy(strategy: BaseStrategy) -> None
#   Append a strategy to the dispatch list; log symbol subscription.
#
# run() -> None
#   Start db + broker, request data for all symbols, enter event loop (blocks).
#
# stop() -> None
#   Set stop event, cancel all orders, disconnect broker, stop db writer.
#
# toggle_strategy(strategy_id: str, active: bool) -> None
#   Enable / disable a strategy at runtime; updates shared_state.
#
# shared_state -> dict   [property]
#   Thread-safe shallow copy of internal state dict.
#   Keys: is_running, is_halted, daily_pnl, peak_equity, drawdown_pct,
#         last_heartbeat, positions, active_strategies, fills_today,
#         error_count_today
#
# risk_manager -> RiskManager   [property]
#   Direct reference for the dashboard kill-switch button.
