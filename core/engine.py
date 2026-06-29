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
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from zoneinfo import ZoneInfo

from config import AppConfig
from core.broker import Broker
from core.events import (
    BarEvent,
    Direction,
    FillEvent,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from database.writer import DatabaseWriter
from notifications.telegram import TelegramNotifier
from risk.manager import RiskManager

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")


@dataclass
class _OpenPosition:
    """Authoritative record of one open position, built from actual fills.

    Used by the engine to match closing fills against their opening fills so
    realised round-trip PnL can be booked into the RiskManager (which drives
    the daily-loss and drawdown kill switches).
    """

    symbol: str
    strategy_id: str
    side: "Direction"          # LONG or SHORT — the side of the position itself
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_commission: float = 0.0


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
            "data_stale": False,
        }
        self._state_lock = threading.Lock()

        # Telegram notifier (no-op if not configured)
        tg = config.telegram
        self._notifier = TelegramNotifier(tg.token, tg.chat_id) if tg.enabled else _NullNotifier()

        # Authoritative position ledger (symbol → open position), built from
        # fills. Drives round-trip PnL accounting and the risk kill switches.
        self._open_positions: dict[str, _OpenPosition] = {}

        # Daily reset tracking
        self._reset_done_today: bool = False
        self._reset_date: object = None  # datetime.date | None

        # Periodic state update tracking
        self._last_periodic_update: float = 0.0

        # Market-data staleness watchdog
        self._last_realtime_bar_mono: float | None = None
        self._data_stale: bool = False
        self._prev_in_rth: bool = False
        # (req_id, symbol, bar_size, duration) for each live subscription
        self._subscriptions: list[tuple[int, str, str, str]] = []
        self._next_req_id: int = 1

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
        self._notifier.notify_engine_start(self._config.is_paper_trading)
        self._notifier.start_daily_summary(self)

        # Request historical (+ live) data for every unique (symbol, bar_size)
        req_id = 1
        seen: set[tuple[str, str]] = set()
        for strategy in self._strategies:
            for symbol in strategy.symbols:
                key = (symbol, strategy.bar_size)
                if key in seen:
                    continue
                seen.add(key)
                duration = self._history_duration(strategy.bar_size)
                self._broker.request_historical_data(
                    req_id, symbol, strategy.bar_size, duration
                )
                self._subscriptions.append(
                    (req_id, symbol, strategy.bar_size, duration)
                )
                req_id += 1
        self._next_req_id = req_id

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
                    if self._strategy_wants(strategy, bar):
                        strategy.on_bar(bar)
                        # Historical bars warm up indicators only — no orders
            elif kind == "realtime":
                self._note_realtime_bar()
                self._db.write(bar)
                for strategy in self._strategies:
                    if self._strategy_wants(strategy, bar):
                        signal = strategy.on_bar(bar)
                        if signal is not None:
                            self._handle_signal(signal, strategy)
        elif isinstance(item, FillEvent):
            self._db.write(item)
            for strategy in self._strategies:
                if strategy.strategy_id == item.strategy_id:
                    strategy.on_fill(item)
            self._risk.record_fill(item)
            self._account_for_fill(item)
            self._update_state_on_fill(item)
            self._notifier.notify_fill(item)
        elif isinstance(item, SystemEvent):
            self._db.write(item)
            if item.kind in (
                SystemEventKind.ENGINE_STOP,
                SystemEventKind.KILL_SWITCH,
            ):
                if item.kind == SystemEventKind.KILL_SWITCH:
                    self._notifier.notify_kill_switch()
                elif "closed" in item.message.lower() or "disconnected" in item.message.lower():
                    self._notifier.notify_connection_lost()
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

        # FLAT: close the open position with a market order (cancels its resting
        # bracket legs first). Sizing/levels do not apply.
        if signal.direction == Direction.FLAT:
            pos = self._open_positions.get(signal.symbol)
            if pos is None:
                self._logger.info(
                    "FLAT signal for %s but no open position to close",
                    signal.symbol,
                )
                return
            self._broker.flatten_symbol(
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                quantity=pos.quantity,
                position_side=pos.side,
            )
            self._logger.info(
                "Flattening %s: %s %d shares",
                signal.symbol, pos.side.name, pos.quantity,
            )
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

    @staticmethod
    def _strategy_wants(strategy: "BaseStrategy", bar: BarEvent) -> bool:
        """True if *bar* should be delivered to *strategy*.

        Must match BOTH symbol and bar size: two strategies can watch the same
        symbol at different bar sizes (e.g. AMZN 15-min for Bollinger and 5-min
        for ORB), and each must only receive its own bar size — otherwise their
        indicator histories cross-contaminate.
        """
        return (
            strategy.is_active
            and bar.symbol in strategy.symbols
            and strategy.bar_size == bar.bar_size
        )

    @staticmethod
    def _history_duration(bar_size: str) -> str:
        """Pick a historical-data window large enough to warm up indicators.

        Critically, 4-hour bars need enough history for long EMAs (the trend
        strategy's 55-period slow EMA needs ~55 RTH bars ≈ many weeks); the old
        flat "5 D" gave only ~10 bars, so that strategy could never warm up or
        trade.
        """
        return {
            "1 day":   "1 Y",
            "4 hours": "6 M",   # ~250 RTH 4H bars — well past a 55-bar warm-up
            "1 hour":  "2 M",
            "30 mins": "1 M",
            "15 mins": "10 D",
            "5 mins":  "5 D",
            "1 min":   "2 D",
        }.get(bar_size, "10 D")

    def _aggregate_positions(self) -> dict[str, int]:
        """Merge per-strategy positions into one combined dict."""
        combined: dict[str, int] = {}
        for strategy in self._strategies:
            for symbol, qty in strategy.positions.items():
                combined[symbol] = combined.get(symbol, 0) + qty
        return combined

    def _account_for_fill(self, fill: FillEvent) -> None:
        """Maintain the position ledger and book realised PnL on round-trips.

        The broker reports a fill's ``direction`` as the *execution* side
        (BOT → LONG, SLD → SHORT), not the strategy's intent. This method pairs
        each closing fill with its opening fill, computes realised PnL, and
        feeds it to the RiskManager so the daily-loss and drawdown kill switches
        actually fire. Runs on the single event-loop thread, so the ledger
        needs no extra lock.
        """
        symbol = fill.symbol
        exec_is_buy = fill.direction == Direction.LONG  # BOT=LONG, SLD=SHORT
        qty = int(fill.quantity)
        if qty <= 0:
            return
        price = float(fill.fill_price)
        commission = float(fill.commission)

        open_pos = self._open_positions.get(symbol)

        # --- Opening a brand-new position --------------------------------
        if open_pos is None:
            self._open_positions[symbol] = _OpenPosition(
                symbol=symbol,
                strategy_id=fill.strategy_id,
                side=Direction.LONG if exec_is_buy else Direction.SHORT,
                quantity=qty,
                entry_price=price,
                entry_time=fill.timestamp,
                entry_commission=commission,
            )
            return

        opening_is_buy = open_pos.side == Direction.LONG

        # --- Adding to the same side: update weighted-average entry -------
        if exec_is_buy == opening_is_buy:
            total_qty = open_pos.quantity + qty
            if total_qty > 0:
                open_pos.entry_price = (
                    open_pos.entry_price * open_pos.quantity + price * qty
                ) / total_qty
            open_pos.quantity = total_qty
            open_pos.entry_commission += commission
            return

        # --- Opposite side: closing (fully or partially) -----------------
        close_qty = min(qty, open_pos.quantity)
        entry_comm_share = (
            open_pos.entry_commission * (close_qty / open_pos.quantity)
            if open_pos.quantity
            else 0.0
        )
        was_halted = self._risk.is_halted

        pnl = self._risk.record_close(
            symbol=symbol,
            entry_price=open_pos.entry_price,
            exit_price=price,
            quantity=close_qty,
            direction=open_pos.side,
            commission=entry_comm_share + commission,
        )
        self._logger.info(
            "Round-trip closed: %s %s qty=%d entry=%.2f exit=%.2f pnl=%.2f",
            open_pos.side.name, symbol, close_qty,
            open_pos.entry_price, price, pnl,
        )

        # Reduce or remove the ledger entry
        open_pos.entry_commission -= entry_comm_share
        open_pos.quantity -= close_qty
        if open_pos.quantity <= 0:
            self._open_positions.pop(symbol, None)
            # Rare: exit qty exceeded the position (a flip) — open the remainder
            remainder = qty - close_qty
            if remainder > 0:
                self._open_positions[symbol] = _OpenPosition(
                    symbol=symbol,
                    strategy_id=fill.strategy_id,
                    side=Direction.LONG if exec_is_buy else Direction.SHORT,
                    quantity=remainder,
                    entry_price=price,
                    entry_time=fill.timestamp,
                    entry_commission=0.0,
                )

        # Surface a newly-triggered automatic halt
        if self._risk.is_halted and not was_halted:
            self._on_risk_halt()

    def _on_risk_halt(self) -> None:
        """Surface an automatic risk-triggered kill switch.

        Alerts via Telegram, writes an audit SystemEvent, and updates dashboard
        state. Deliberately does NOT call stop(): new entries are blocked (via
        RiskManager.is_halted), but resting bracket stops/targets stay live to
        keep protecting any open positions.
        """
        msg = (
            f"Automatic kill switch tripped: "
            f"daily_pnl={self._risk.daily_pnl:.2f} "
            f"drawdown={self._risk.current_drawdown_pct:.1%}. "
            f"New entries blocked; resting stops/targets remain active."
        )
        self._logger.critical(msg)
        try:
            self._db.write(
                SystemEvent(
                    kind=SystemEventKind.KILL_SWITCH,
                    timestamp=datetime.now(_ET),
                    message=msg,
                )
            )
        except Exception:
            self._logger.exception("Failed to persist kill-switch event")
        with self._state_lock:
            self._state["is_halted"] = True
        try:
            self._notifier.notify_kill_switch()
        except Exception:
            self._logger.exception("Failed to send kill-switch notification")

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

    # ------------------------------------------------------------------
    # Market-data staleness watchdog
    # ------------------------------------------------------------------

    def _note_realtime_bar(self) -> None:
        """Record that a live bar arrived; clear any active staleness alert."""
        self._last_realtime_bar_mono = time.monotonic()
        if self._data_stale:
            self._data_stale = False
            with self._state_lock:
                self._state["data_stale"] = False
            self._logger.warning("Market data restored — live bars flowing again")
            try:
                self._notifier.notify_data_restored()
            except Exception:
                self._logger.exception("notify_data_restored failed")

    def _is_rth(self, now_et: datetime) -> bool:
        """True if now_et is within regular US equity trading hours (Mon–Fri).

        Holiday-unaware: a market holiday would at worst produce one spurious
        stale-alert, which is acceptable for a safety watchdog.
        """
        if now_et.weekday() >= 5:                 # Saturday / Sunday
            return False
        md = self._config.market_data
        start = now_et.replace(hour=md.rth_start[0], minute=md.rth_start[1],
                               second=0, microsecond=0)
        end = now_et.replace(hour=md.rth_end[0], minute=md.rth_end[1],
                             second=0, microsecond=0)
        return start <= now_et < end

    def _check_data_staleness(self, now_et: "datetime | None" = None,
                              now_mono: "float | None" = None) -> None:
        """Alert if no live bar has arrived for too long during trading hours.

        Re-arms each session. ``now_et`` / ``now_mono`` are injectable for tests.
        """
        md = self._config.market_data
        if not md.watchdog_enabled:
            return
        now_et = now_et if now_et is not None else datetime.now(_ET)
        now_mono = now_mono if now_mono is not None else time.monotonic()

        in_rth = self._is_rth(now_et)
        # On the open transition, (re)start the clock and clear any prior alert.
        if in_rth and not self._prev_in_rth:
            self._last_realtime_bar_mono = now_mono
            self._data_stale = False
            with self._state_lock:
                self._state["data_stale"] = False
        self._prev_in_rth = in_rth
        if not in_rth:
            return

        if self._last_realtime_bar_mono is None:
            self._last_realtime_bar_mono = now_mono
            return
        elapsed = now_mono - self._last_realtime_bar_mono
        if elapsed >= md.staleness_timeout_seconds and not self._data_stale:
            self._data_stale = True
            self._on_data_stale(int(elapsed // 60))

    def _on_data_stale(self, minutes: int) -> None:
        """Fire the loud alert for a market-data blackout (once per episode)."""
        msg = (
            f"DATA STALE: no live bar in ~{minutes} min during RTH. Engine "
            f"connected but receiving no market data — strategies are blind and "
            f"no orders will fire. Check TWS data subscription / data farm."
        )
        self._logger.error(msg)
        with self._state_lock:
            self._state["data_stale"] = True
            self._state["error_count_today"] += 1
        try:
            self._db.write(SystemEvent(
                kind=SystemEventKind.ERROR,
                timestamp=datetime.now(_ET),
                message=msg,
            ))
        except Exception:
            self._logger.exception("Failed to persist data-stale event")
        try:
            self._notifier.notify_data_stale(minutes)
        except Exception:
            self._logger.exception("notify_data_stale failed")
        if self._config.market_data.resubscribe_on_stale:
            self._resubscribe_market_data()

    def _resubscribe_market_data(self) -> None:
        """Best-effort: cancel and re-request all keepUpToDate subscriptions.

        Uses fresh req_ids to avoid colliding with a possibly-stuck id. Never
        raises — the watchdog must not be able to crash the event loop.
        """
        try:
            old = list(self._subscriptions)
            self._subscriptions = []
            for req_id, symbol, bar_size, duration in old:
                self._broker.cancel_historical_data(req_id)
                new_id = self._next_req_id
                self._next_req_id += 1
                self._broker.request_historical_data(new_id, symbol, bar_size, duration)
                self._subscriptions.append((new_id, symbol, bar_size, duration))
            self._logger.warning("Re-requested %d market-data subscriptions", len(old))
        except Exception:
            self._logger.exception("Market-data re-subscribe failed")

    def _maybe_update_state_periodic(self) -> None:
        """Refresh shared state approximately every 5 seconds."""
        now = time.monotonic()
        if now - self._last_periodic_update >= 5.0:
            self._update_state_periodic()
            self._check_data_staleness()
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


class _NullNotifier:
    """No-op notifier used when Telegram is not configured."""
    def notify_fill(self, fill: object) -> None: pass
    def notify_kill_switch(self) -> None: pass
    def notify_connection_lost(self) -> None: pass
    def notify_connection_restored(self) -> None: pass
    def notify_engine_start(self, paper: bool) -> None: pass
    def notify_data_stale(self, minutes: int) -> None: pass
    def notify_data_restored(self) -> None: pass
    def start_daily_summary(self, engine: object) -> None: pass


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
