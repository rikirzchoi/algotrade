"""
core.events — canonical event dataclasses flowing through the trading engine.

Every piece of information passed between the broker, engine, strategies, risk
manager, and database travels as one of these typed dataclasses.  Using a
shared, immutable contract here means each module can be developed and tested
in isolation.

Event hierarchy
---------------
BarEvent        — a completed OHLCV bar for one symbol
SignalEvent     — a strategy's intent to go long / short / flat
OrderEvent      — an order that the engine has decided to send to the broker
FillEvent       — a confirmed execution returned by the broker
SystemEvent     — lifecycle notifications (start, stop, error, kill-switch)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional


class EventType(enum.Enum):
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    SYSTEM = "SYSTEM"


class Direction(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(enum.Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    BRACKET = "BRACKET"


class SystemEventKind(enum.Enum):
    ENGINE_START = "ENGINE_START"
    ENGINE_STOP = "ENGINE_STOP"
    KILL_SWITCH = "KILL_SWITCH"
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class BarEvent:
    """A single completed OHLCV bar delivered to all registered strategies."""

    event_type: EventType = field(default=EventType.BAR, init=False)
    symbol: str
    timestamp: datetime          # America/New_York
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    bar_size: str                # e.g. "1 min", "5 mins", "1 day"
    vwap: Optional[Decimal] = None


@dataclass(frozen=True)
class SignalEvent:
    """Intent emitted by a strategy; the engine decides whether to act on it."""

    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    strategy_id: str
    symbol: str
    timestamp: datetime
    direction: Direction
    strength: float              # normalised 0.0–1.0; used for sizing
    reason: str = ""             # human-readable explanation for logs/dashboard


@dataclass(frozen=True)
class OrderEvent:
    """An order the engine has approved and is about to send to the broker."""

    event_type: EventType = field(default=EventType.ORDER, init=False)
    strategy_id: str
    symbol: str
    timestamp: datetime
    direction: Direction
    order_type: OrderType
    quantity: int
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    client_order_id: Optional[int] = None


@dataclass(frozen=True)
class FillEvent:
    """A confirmed execution received from Interactive Brokers."""

    event_type: EventType = field(default=EventType.FILL, init=False)
    strategy_id: str
    symbol: str
    timestamp: datetime
    direction: Direction
    quantity: int
    fill_price: Decimal
    commission: Decimal
    client_order_id: Optional[int] = None
    ib_exec_id: str = ""


@dataclass(frozen=True)
class SystemEvent:
    """Lifecycle and error notifications emitted by the engine or broker."""

    event_type: EventType = field(default=EventType.SYSTEM, init=False)
    kind: SystemEventKind
    timestamp: datetime
    message: str = ""
    details: Optional[dict] = None
