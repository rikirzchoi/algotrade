"""
database.writer — the ONLY module that writes to the SQLite database.

Creates and maintains five tables:

    bars      — every OHLCV bar received (used for charting + replay)
    signals   — every SignalEvent emitted by strategies
    orders    — every OrderEvent approved by the risk manager
    fills     — every FillEvent confirmed by the broker
    system    — SystemEvents (start/stop/errors/kill-switch activations)

All writes are batched and flushed on a configurable interval (default 1 s)
to avoid blocking the engine's event loop.  A dedicated writer thread owns
the SQLite connection so that no connection is shared across threads.

Schema migrations are handled with a simple version table; the writer creates
missing tables automatically on first run.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Union

from core.events import BarEvent, FillEvent, OrderEvent, SignalEvent, SystemEvent

logger = logging.getLogger(__name__)

# Type alias for any writable event
WritableEvent = Union[BarEvent, SignalEvent, OrderEvent, FillEvent, SystemEvent]


class DatabaseWriter:
    """
    Asynchronous SQLite writer.

    Events are placed on an internal queue by the engine thread and flushed
    to disk by a dedicated background thread, keeping write latency off the
    critical path.
    """

    def __init__(self, db_path: Path, flush_interval_seconds: float = 1.0) -> None:
        self._db_path = Path(db_path)
        self._flush_interval = flush_interval_seconds
        self._queue: queue.Queue[WritableEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the database connection and start the background flush thread."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()
        logger.info("DatabaseWriter started (db=%s)", self._db_path)

    def stop(self) -> None:
        """Flush remaining events, close the connection, and stop the thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("DatabaseWriter stopped")

    # ------------------------------------------------------------------
    # Public write API (thread-safe; called from engine thread)
    # ------------------------------------------------------------------

    def write(self, event: WritableEvent) -> None:
        """Place *event* on the write queue for async persistence."""
        self._queue.put_nowait(event)

    # ------------------------------------------------------------------
    # Private helpers (called from writer thread only)
    # ------------------------------------------------------------------

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create all tables if they do not exist; run any pending migrations."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT OR IGNORE INTO schema_version VALUES (1);

            CREATE TABLE IF NOT EXISTS bars (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                open      REAL    NOT NULL,
                high      REAL    NOT NULL,
                low       REAL    NOT NULL,
                close     REAL    NOT NULL,
                volume    INTEGER NOT NULL,
                bar_size  TEXT    NOT NULL,
                vwap      REAL
            );
            CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, timestamp);

            CREATE TABLE IF NOT EXISTS signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                direction   TEXT    NOT NULL,
                strength    REAL    NOT NULL,
                reason      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals(strategy_id, timestamp);

            CREATE TABLE IF NOT EXISTS orders (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id       TEXT    NOT NULL,
                symbol            TEXT    NOT NULL,
                timestamp         TEXT    NOT NULL,
                direction         TEXT    NOT NULL,
                order_type        TEXT    NOT NULL,
                quantity          INTEGER NOT NULL,
                limit_price       REAL,
                stop_price        REAL,
                take_profit_price REAL,
                client_order_id   INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_orders_strategy_ts ON orders(strategy_id, timestamp);

            CREATE TABLE IF NOT EXISTS fills (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id     TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL,
                direction       TEXT    NOT NULL,
                quantity        INTEGER NOT NULL,
                fill_price      REAL    NOT NULL,
                commission      REAL    NOT NULL,
                client_order_id INTEGER,
                ib_exec_id      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_fills_strategy_ts ON fills(strategy_id, timestamp);

            CREATE TABLE IF NOT EXISTS system (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                kind      TEXT    NOT NULL,
                timestamp TEXT    NOT NULL,
                message   TEXT,
                details   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_system_ts ON system(timestamp);
        """)
        conn.commit()

    def _flush(self, conn: sqlite3.Connection) -> None:
        """Drain the queue and INSERT all pending events in a single transaction."""
        events: list[WritableEvent] = []
        try:
            while True:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not events:
            return

        try:
            with conn:
                for event in events:
                    if isinstance(event, BarEvent):
                        self._insert_bar(conn, event)
                    elif isinstance(event, SignalEvent):
                        self._insert_signal(conn, event)
                    elif isinstance(event, OrderEvent):
                        self._insert_order(conn, event)
                    elif isinstance(event, FillEvent):
                        self._insert_fill(conn, event)
                    elif isinstance(event, SystemEvent):
                        self._insert_system(conn, event)
                    else:
                        logger.warning("Unknown event type skipped: %s", type(event))
        except sqlite3.Error:
            logger.exception("Error flushing %d events to database", len(events))

    def _run(self) -> None:
        """Background thread: periodically flush the event queue to SQLite."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            logger.exception("Failed to open database %s", self._db_path)
            return

        try:
            self._create_schema(conn)
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=self._flush_interval)
                self._flush(conn)
            # Final flush to capture any events queued before stop()
            self._flush(conn)
        except Exception:
            logger.exception("Unexpected error in database writer thread")
        finally:
            conn.close()

    def _insert_bar(self, conn: sqlite3.Connection, event: BarEvent) -> None:
        conn.execute(
            """INSERT INTO bars (symbol, timestamp, open, high, low, close, volume, bar_size, vwap)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.symbol,
                event.timestamp.isoformat(),
                float(event.open),
                float(event.high),
                float(event.low),
                float(event.close),
                event.volume,
                event.bar_size,
                float(event.vwap) if event.vwap is not None else None,
            ),
        )

    def _insert_signal(self, conn: sqlite3.Connection, event: SignalEvent) -> None:
        conn.execute(
            """INSERT INTO signals (strategy_id, symbol, timestamp, direction, strength, reason)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event.strategy_id,
                event.symbol,
                event.timestamp.isoformat(),
                event.direction.value,
                event.strength,
                event.reason,
            ),
        )

    def _insert_order(self, conn: sqlite3.Connection, event: OrderEvent) -> None:
        conn.execute(
            """INSERT INTO orders (strategy_id, symbol, timestamp, direction, order_type,
                                   quantity, limit_price, stop_price, take_profit_price,
                                   client_order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.strategy_id,
                event.symbol,
                event.timestamp.isoformat(),
                event.direction.value,
                event.order_type.value,
                event.quantity,
                float(event.limit_price) if event.limit_price is not None else None,
                float(event.stop_price) if event.stop_price is not None else None,
                float(event.take_profit_price) if event.take_profit_price is not None else None,
                event.client_order_id,
            ),
        )

    def _insert_fill(self, conn: sqlite3.Connection, event: FillEvent) -> None:
        conn.execute(
            """INSERT INTO fills (strategy_id, symbol, timestamp, direction, quantity,
                                  fill_price, commission, client_order_id, ib_exec_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.strategy_id,
                event.symbol,
                event.timestamp.isoformat(),
                event.direction.value,
                event.quantity,
                float(event.fill_price),
                float(event.commission),
                event.client_order_id,
                event.ib_exec_id,
            ),
        )

    def _insert_system(self, conn: sqlite3.Connection, event: SystemEvent) -> None:
        conn.execute(
            """INSERT INTO system (kind, timestamp, message, details)
               VALUES (?, ?, ?, ?)""",
            (
                event.kind.value,
                event.timestamp.isoformat(),
                event.message,
                json.dumps(event.details) if event.details is not None else None,
            ),
        )
