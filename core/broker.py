"""
broker.py — the ONLY file in the codebase that imports or touches ibapi.

All other modules communicate through the event queue and public methods of
the Broker class.  ibapi callbacks are converted to typed event dataclasses
(defined in core.events) before being placed on the queue.

Usage::

    from config import AppConfig
    from broker import Broker

    cfg = AppConfig.from_env()
    broker = Broker(cfg)
    broker.connect()

    # Engine reads from this queue in a tight loop
    event = broker.event_queue.get()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.execution import Execution
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from config import AppConfig
from core.events import (
    BarEvent,
    Direction,
    FillEvent,
    SystemEvent,
    SystemEventKind,
)

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Informational / connectivity codes that should never surface as errors
# ---------------------------------------------------------------------------

_IGNORE_CODES: frozenset[int] = frozenset(
    {2100, 2101, 2102, 2103, 2104, 2106, 2107, 2108, 2158}
)
_WARN_CODES: frozenset[int] = frozenset({200, 354})

_HEARTBEAT_INTERVAL = 30  # seconds


class Broker(EWrapper, EClient):
    """Bridge between Interactive Brokers (ibapi) and the trading engine.

    Connects to TWS / IB Gateway, requests market data, places and cancels
    orders, and converts all IBKR callbacks into typed event dataclasses that
    are placed on the internal event queue for the engine to consume.

    Threading model
    ---------------
    * EClient.run() spins in a dedicated daemon thread.
    * A separate heartbeat daemon thread emits periodic SystemEvents.
    * All shared mutable state is protected with threading.Lock or
      threading.Event as appropriate.
    * The public event_queue is thread-safe (queue.Queue).
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialise internal state.  Does NOT connect to IB."""
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self._config = config

        # Queue that the engine drains; items are (str, EventType) tuples
        # e.g. ("historical", BarEvent(...)) or bare SystemEvent / FillEvent.
        self._event_queue: queue.Queue = queue.Queue()

        self._order_id_lock = threading.Lock()
        self._next_order_id: int = 0

        # Set by nextValidId — connect() blocks until this fires or times out
        self._connected = threading.Event()

        # reqId → symbol / bar_size mappings for callback reconstruction
        self._req_id_to_symbol: dict[int, str] = {}
        self._req_id_to_bar_size: dict[int, str] = {}

        # client_order_id (str) → strategy_id for routing fills
        self._strategy_order_map: dict[str, str] = {}

        # Placeholder; assigned in connect()
        self._heartbeat_thread: threading.Thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="broker-heartbeat"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def event_queue(self) -> queue.Queue:
        """Read-only access to the event queue for the engine."""
        return self._event_queue

    def connect(self) -> None:  # noqa: D102
        """Connect to IB TWS / Gateway and start the message loop.

        Blocks up to 10 seconds waiting for the API to confirm a valid order
        ID (nextValidId callback), which signals a fully established session.

        Raises:
            ConnectionError: if the connection is not confirmed within 10 s.
        """
        cfg = self._config.ibkr
        log.info(
            "Connecting to IB on %s:%s (client_id=%s)", cfg.host, cfg.port, cfg.client_id
        )
        EClient.connect(self, cfg.host, cfg.port, cfg.client_id)

        api_thread = threading.Thread(
            target=self.run, daemon=True, name="broker-api"
        )
        api_thread.start()

        if not self._connected.wait(timeout=10.0):
            raise ConnectionError(
                f"IB connection timed out after 10 s "
                f"({cfg.host}:{cfg.port} client_id={cfg.client_id})"
            )

        # Start heartbeat (emits ENGINE_START immediately, then every 30 s)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="broker-heartbeat"
        )
        self._heartbeat_thread.start()
        log.info("IB connection established.")

    def disconnect(self) -> None:
        """Disconnect from IB and signal the engine to stop."""
        log.info("Disconnecting from IB.")
        EClient.disconnect(self)
        self._event_queue.put(
            SystemEvent(
                kind=SystemEventKind.ENGINE_STOP,
                timestamp=datetime.now(tz=ET),
                message="Broker disconnected.",
            )
        )

    def request_historical_data(
        self,
        req_id: int,
        symbol: str,
        bar_size: str,
        duration: str,
    ) -> None:
        """Request historical bars and keep the subscription live for realtime updates.

        Args:
            req_id:   Unique request identifier (caller's responsibility to
                      ensure uniqueness).
            symbol:   Ticker symbol (e.g. "SPY").
            bar_size: IBKR bar size string (e.g. "1 day", "15 mins", "1 min").
            duration: IBKR duration string (e.g. "2 D", "1 Y").
        """
        self._req_id_to_symbol[req_id] = symbol
        self._req_id_to_bar_size[req_id] = bar_size

        contract = self._make_contract(symbol)

        log.debug(
            "reqHistoricalData req_id=%s symbol=%s bar_size=%s duration=%s",
            req_id, symbol, bar_size, duration,
        )
        self.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",        # empty = now
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=True,     # streams realtime bars after history is done
            chartOptions=[],
        )

    def place_bracket_order(
        self,
        client_order_id: str,
        strategy_id: str,
        symbol: str,
        direction: Direction,
        quantity: int,
        entry_price: float,
        profit_target: float,
        stop_loss: float,
    ) -> None:
        """Submit a bracket order (entry + profit target + stop loss) to IB.

        All three legs are transmitted atomically: the parent and profit-target
        legs have transmit=False; the stop-loss leg has transmit=True, which
        causes IB to send all three at once.

        Args:
            client_order_id: Caller-supplied identifier stored in
                             _strategy_order_map to route fills back.
            strategy_id:     Strategy that generated the signal.
            symbol:          Ticker symbol.
            direction:       Direction.LONG or Direction.SHORT.
            quantity:        Number of shares.
            entry_price:     Reference entry price (used for logging; the
                             parent order is MKT so IB ignores the price).
            profit_target:   Limit price for the take-profit leg.
            stop_loss:       Stop price for the stop-loss leg.
        """
        parent_id = self.get_next_order_id()
        profit_id = self.get_next_order_id()
        stop_id = self.get_next_order_id()

        is_long = direction == Direction.LONG
        entry_action = "BUY" if is_long else "SELL"
        exit_action = "SELL" if is_long else "BUY"

        contract = self._make_contract(symbol)

        # --- Parent (market entry) ---
        parent = Order()
        parent.orderId = parent_id
        parent.action = entry_action
        parent.orderType = "MKT"
        parent.totalQuantity = quantity
        parent.transmit = False

        # --- Profit target (limit) ---
        take_profit = Order()
        take_profit.orderId = profit_id
        take_profit.action = exit_action
        take_profit.orderType = "LMT"
        take_profit.totalQuantity = quantity
        take_profit.lmtPrice = round(profit_target, 2)
        take_profit.parentId = parent_id
        take_profit.transmit = False

        # --- Stop loss (stop) ---
        stop = Order()
        stop.orderId = stop_id
        stop.action = exit_action
        stop.orderType = "STP"
        stop.totalQuantity = quantity
        stop.auxPrice = round(stop_loss, 2)
        stop.parentId = parent_id
        stop.transmit = True  # transmits all three at once

        # Register mapping so fills can be routed to the correct strategy
        self._strategy_order_map[str(parent_id)] = strategy_id
        self._strategy_order_map[str(profit_id)] = strategy_id
        self._strategy_order_map[str(stop_id)] = strategy_id
        # Also store by the caller's logical ID
        self._strategy_order_map[client_order_id] = strategy_id

        log.info(
            "Placing bracket order: symbol=%s direction=%s qty=%d "
            "entry≈%.2f target=%.2f stop=%.2f [ids %d/%d/%d]",
            symbol, direction.value, quantity,
            entry_price, profit_target, stop_loss,
            parent_id, profit_id, stop_id,
        )
        self.placeOrder(parent_id, contract, parent)
        self.placeOrder(profit_id, contract, take_profit)
        self.placeOrder(stop_id, contract, stop)

    def cancel_order(self, order_id: int) -> None:
        """Cancel an existing order by its IB order ID."""
        log.info("Cancelling order id=%d", order_id)
        EClient.cancelOrder(self, order_id, "")

    def cancel_all_orders(self) -> None:
        """Cancel every open order via IB's global cancel request."""
        log.info("Cancelling all open orders (reqGlobalCancel).")
        self.reqGlobalCancel()

    def get_next_order_id(self) -> int:
        """Return the next available order ID and increment the counter.

        Thread-safe: protected by _order_id_lock.
        """
        with self._order_id_lock:
            oid = self._next_order_id
            self._next_order_id += 1
            return oid

    # ------------------------------------------------------------------
    # EWrapper callbacks
    # ------------------------------------------------------------------

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        """Called by IB when the session is ready; seeds the order ID counter."""
        super().nextValidId(orderId)
        with self._order_id_lock:
            self._next_order_id = orderId
        log.debug("nextValidId: first order id = %d", orderId)
        self._connected.set()

    def historicalData(self, reqId: int, bar) -> None:  # noqa: N802
        """Deliver a single historical bar; wraps it as ("historical", BarEvent)."""
        try:
            event = self._bar_event_from_ibkr_bar(reqId, bar)
            self._event_queue.put(("historical", event))
        except Exception:
            log.exception("historicalData: failed to build BarEvent for reqId=%d", reqId)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
        """Called when the backfill phase of a keepUpToDate subscription ends."""
        log.info("Historical data complete for reqId %d (%s → %s)", reqId, start, end)
        self._event_queue.put(
            SystemEvent(
                kind=SystemEventKind.WARNING,
                timestamp=datetime.now(tz=ET),
                message=f"Historical data complete for reqId {reqId}",
                details={"req_id": reqId, "start": start, "end": end},
            )
        )

    def historicalDataUpdate(self, reqId: int, bar) -> None:  # noqa: N802
        """Deliver a live bar from a keepUpToDate subscription; wraps as ("realtime", BarEvent)."""
        try:
            event = self._bar_event_from_ibkr_bar(reqId, bar)
            self._event_queue.put(("realtime", event))
        except Exception:
            log.exception(
                "historicalDataUpdate: failed to build BarEvent for reqId=%d", reqId
            )

    def realtimeBar(  # noqa: N802
        self,
        reqId: int,
        time: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        wap: float,
        count: int,
    ) -> None:
        """Deliver a 5-second realtime bar from reqRealTimeBars.

        Constructs a proper BarEvent from the raw numeric values; does NOT
        pass raw values through unchanged.
        """
        try:
            ts = datetime.fromtimestamp(time, tz=ET)
            bar_size = self._req_id_to_bar_size.get(reqId, "5 secs")
            symbol = self._req_id_to_symbol.get(reqId, "")
            event = BarEvent(
                symbol=symbol,
                timestamp=ts,
                open=Decimal(str(open_)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=int(volume),
                bar_size=bar_size,
                vwap=Decimal(str(wap)) if wap else None,
            )
            self._event_queue.put(("realtime", event))
        except Exception:
            log.exception("realtimeBar: failed to build BarEvent for reqId=%d", reqId)

    def execDetails(  # noqa: N802
        self, reqId: int, contract: Contract, execution: Execution
    ) -> None:
        """Convert an execution report into a FillEvent and queue it."""
        try:
            strategy_id = self._strategy_order_map.get(
                str(execution.orderId), "unknown"
            )
            direction = (
                Direction.LONG if execution.side == "BOT" else Direction.SHORT
            )
            # Parse IB timestamp format "YYYYMMDD  HH:MM:SS" (note double space)
            try:
                ts = datetime.strptime(
                    execution.time.strip(), "%Y%m%d  %H:%M:%S"
                ).replace(tzinfo=ET)
            except ValueError:
                ts = datetime.now(tz=ET)

            fill = FillEvent(
                strategy_id=strategy_id,
                symbol=contract.symbol,
                timestamp=ts,
                direction=direction,
                quantity=int(execution.shares),
                fill_price=Decimal(str(execution.price)),
                commission=Decimal("0.0"),  # updated in commissionReport callback
                client_order_id=execution.orderId,
                ib_exec_id=execution.execId,
            )
            self._event_queue.put(fill)
        except Exception:
            log.exception("execDetails: failed to build FillEvent")

    def commissionReport(self, commissionReport) -> None:  # noqa: N802
        """Called after execDetails with actual commission data.

        TODO: Update the corresponding FillEvent with the real commission.
              This requires a pending-fills store keyed by execId so the
              commission can be patched in and the corrected FillEvent
              re-queued (or the engine can handle it as a separate event).
              For now we log it so it is not silently discarded.
        """
        log.info(
            "commissionReport: execId=%s commission=%.4f currency=%s",
            commissionReport.execId,
            commissionReport.commission,
            commissionReport.currency,
        )

    def error(  # noqa: N802
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        """Route IB error/info codes to the appropriate log level or queue."""
        if errorCode in _IGNORE_CODES:
            log.debug("IB info [%d] reqId=%d: %s", errorCode, reqId, errorString)
            return

        if errorCode in _WARN_CODES:
            log.warning("IB warning [%d] reqId=%d: %s", errorCode, reqId, errorString)
            return

        details: dict = {
            "req_id": reqId,
            "error_code": errorCode,
            "advanced_order_reject": advancedOrderRejectJson or None,
        }

        if errorCode >= 400:
            log.error("IB error [%d] reqId=%d: %s", errorCode, reqId, errorString)
            self._event_queue.put(
                SystemEvent(
                    kind=SystemEventKind.ERROR,
                    timestamp=datetime.now(tz=ET),
                    message=f"IB error {errorCode}: {errorString}",
                    details=details,
                )
            )
            return

        if errorCode < 100 and reqId == -1:
            # Connectivity-level problem
            log.error(
                "IB connectivity error [%d]: %s", errorCode, errorString
            )
            self._event_queue.put(
                SystemEvent(
                    kind=SystemEventKind.ERROR,
                    timestamp=datetime.now(tz=ET),
                    message=f"IB connectivity error {errorCode}: {errorString}",
                    details=details,
                )
            )
            # Ensure connect() doesn't hang forever
            if not self._connected.is_set():
                self._connected.set()
            return

        log.warning("IB message [%d] reqId=%d: %s", errorCode, reqId, errorString)

    def connectionClosed(self) -> None:  # noqa: N802
        """Called by ibapi when the socket is closed by the IB side."""
        log.warning("IB connection closed.")
        self._event_queue.put(
            SystemEvent(
                kind=SystemEventKind.ENGINE_STOP,
                timestamp=datetime.now(tz=ET),
                message="Connection closed by IB",
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_contract(symbol: str) -> Contract:
        """Build a vanilla US equity Contract for the given symbol."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def _bar_event_from_ibkr_bar(self, req_id: int, bar) -> BarEvent:
        """Convert an ibapi bar object to a BarEvent.

        Handles two date formats from IBKR:
        * Daily bars: "YYYYMMDD"
        * Intraday bars: "YYYYMMDD HH:MM:SS" (with timezone suffix dropped)
        """
        symbol = self._req_id_to_symbol.get(req_id, "")
        bar_size = self._req_id_to_bar_size.get(req_id, "")
        raw_date: str = bar.date

        # Intraday timestamps carry a time component; daily bars do not
        if len(raw_date) > 8:
            # Strip optional " US/Eastern" or " America/New_York" suffix that
            # IBKR sometimes appends, then parse
            date_part = raw_date.split(" ")[0] + " " + raw_date.split(" ")[1]
            ts = datetime.strptime(date_part, "%Y%m%d %H:%M:%S").replace(tzinfo=ET)
        else:
            # Daily bar — represent as midnight ET for consistency
            d = datetime.strptime(raw_date, "%Y%m%d")
            ts = d.replace(tzinfo=ET)

        vwap_raw = getattr(bar, "vwap", None) or getattr(bar, "average", None)
        vwap: Optional[Decimal] = (
            Decimal(str(vwap_raw)) if vwap_raw not in (None, -1, -1.0) else Decimal(str(bar.close))
        )

        return BarEvent(
            symbol=symbol,
            timestamp=ts,
            open=Decimal(str(bar.open)),
            high=Decimal(str(bar.high)),
            low=Decimal(str(bar.low)),
            close=Decimal(str(bar.close)),
            volume=int(bar.volume),
            bar_size=bar_size,
            vwap=vwap,
        )

    def _heartbeat_loop(self) -> None:
        """Daemon thread: emit ENGINE_START once, then a WARNING every 30 s."""
        self._event_queue.put(
            SystemEvent(
                kind=SystemEventKind.ENGINE_START,
                timestamp=datetime.now(tz=ET),
                message="Broker connected and ready.",
            )
        )
        while True:
            time.sleep(_HEARTBEAT_INTERVAL)
            if not self.isConnected():
                break
            self._event_queue.put(
                SystemEvent(
                    kind=SystemEventKind.WARNING,
                    timestamp=datetime.now(tz=ET),
                    message="heartbeat",
                )
            )


# ---------------------------------------------------------------------------
# Public method reference card
# ---------------------------------------------------------------------------
#
# Broker(config: AppConfig)
#
# connect() -> None
#   Connect to IB TWS/Gateway; raises ConnectionError after 10 s timeout.
#
# disconnect() -> None
#   Disconnect and enqueue ENGINE_STOP.
#
# request_historical_data(req_id: int, symbol: str,
#                         bar_size: str, duration: str) -> None
#   Subscribe to historical + live bars via keepUpToDate=True.
#
# place_bracket_order(client_order_id: str, strategy_id: str,
#                     symbol: str, direction: Direction,
#                     quantity: int, entry_price: float,
#                     profit_target: float, stop_loss: float) -> None
#   Send atomic bracket (MKT entry + LMT target + STP stop).
#
# cancel_order(order_id: int) -> None
#   Cancel an order by its IB order ID.
#
# cancel_all_orders() -> None
#   Cancel every open order via reqGlobalCancel.
#
# get_next_order_id() -> int
#   Thread-safe counter; returns current ID and increments.
#
# event_queue -> queue.Queue   [property]
#   Engine reads: items are bare SystemEvent/FillEvent or
#   ("historical"|"realtime", BarEvent) tuples.
