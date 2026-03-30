"""
strategies.base — abstract base class that every strategy must implement.

Enforces a consistent interface so the engine can treat all strategies
identically.  Concrete strategies subclass BaseStrategy, declare which symbols
and bar sizes they need, and implement on_bar().  They must never place orders
directly — instead they return or emit a SignalEvent and let the engine route
it through the risk manager.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pytz

from config import AppConfig
from core.data import BarHistory
from core.events import BarEvent, Direction, FillEvent, SignalEvent, SystemEvent

_ET = pytz.timezone("America/New_York")

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.

    Subclasses must implement:
      - on_bar()    — called for every completed bar on a subscribed symbol
      - on_fill()   — called when one of this strategy's orders is filled

    Subclasses may optionally override:
      - on_start()  — called once when the engine starts
      - on_stop()   — called once before the engine shuts down
      - on_system() — called for SystemEvents (e.g. session open/close)
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: list[str],
        config: AppConfig,
    ) -> None:
        self._strategy_id = strategy_id
        self._symbols = list(symbols)
        self._config = config
        self._is_active: bool = True
        self._positions: dict[str, int] = {}
        self._bar_histories: dict[str, BarHistory] = {
            sym: BarHistory(sym) for sym in symbols
        }
        self._logger = logging.getLogger(f"strategy.{strategy_id}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether this strategy is currently accepting bar events."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._is_active = value
        state = "activated" if value else "deactivated"
        self._logger.info(f"Strategy {self._strategy_id} {state}")

    @property
    def strategy_id(self) -> str:
        """Unique identifier used in logs, database records, and the dashboard."""
        return self._strategy_id

    @property
    def symbols(self) -> list[str]:
        """List of ticker symbols this strategy subscribes to."""
        return self._symbols

    @property
    @abstractmethod
    def bar_size(self) -> str:
        """Bar granularity required, e.g. '1 min', '5 mins', '1 day'."""

    @property
    def positions(self) -> dict[str, int]:
        """Current positions by symbol. Returns a copy — not the internal dict."""
        return dict(self._positions)

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def get_history(self, symbol: str) -> BarHistory:
        """Return the BarHistory for *symbol*, auto-creating it if needed."""
        if symbol not in self._bar_histories:
            self._bar_histories[symbol] = BarHistory(symbol)
        return self._bar_histories[symbol]

    def has_position(self, symbol: str) -> bool:
        """Return True if this strategy holds a non-zero position in *symbol*."""
        return self._positions.get(symbol, 0) != 0

    def get_position(self, symbol: str) -> int:
        """Return current quantity. Positive = long, negative = short, 0 = flat."""
        return self._positions.get(symbol, 0)

    def update_position(self, symbol: str, qty_delta: int) -> None:
        """Add *qty_delta* to the current position for *symbol*."""
        current = self._positions.get(symbol, 0)
        updated = current + qty_delta
        self._positions[symbol] = updated
        self._logger.debug(
            "Position update %s: %d -> %d", symbol, current, updated
        )

    def flatten_position(self, symbol: str) -> Optional[SignalEvent]:
        """
        Return a FLAT SignalEvent for *symbol* if a position is open, else None.

        Strategies call this for end-of-day flattening.
        """
        if not self.has_position(symbol):
            return None
        return SignalEvent(
            strategy_id=self._strategy_id,
            symbol=symbol,
            timestamp=datetime.now(_ET),
            direction=Direction.FLAT,
            strength=1.0,
            reason="flatten",
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Called once when the engine starts.  Load any warm-up state here."""

    def on_stop(self) -> None:
        """Called once before the engine shuts down.  Flush state if needed."""

    def on_system(self, event: SystemEvent) -> None:
        """Handle engine-level system notifications (session open/close, errors)."""

    # ------------------------------------------------------------------
    # Core callbacks — must be implemented by every concrete strategy
    # ------------------------------------------------------------------

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Process a completed bar.

        Must catch all exceptions internally and log them — never propagate.
        Return a SignalEvent to express a trade intent, or None for no action.
        """

    @abstractmethod
    def on_fill(self, fill: FillEvent) -> None:
        """
        Called when one of this strategy's orders has been filled.

        Use this to update internal position tracking or adjust stop levels.
        Must never raise.
        """
