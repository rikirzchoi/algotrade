"""
backtesting.costs — a single, shared transaction-cost model.

Used by both the single-symbol BacktestEngine and the PortfolioBacktestEngine
so cost assumptions can't drift apart between them. All percentages are
fractions (0.0005 == 5 bps).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.events import Direction


@dataclass
class CostModel:
    """Conservative execution-cost assumptions for backtesting.

    Components
    ----------
    commission_per_share : flat per-share commission (IBKR-style).
    slippage_pct         : market-impact / timing slippage on every fill.
    spread_pct           : full bid-ask spread; HALF is paid per market fill.
    borrow_rate_annual   : annualised financing rate charged on short positions.
    """

    commission_per_share: float = 0.005
    slippage_pct: float = 0.0005
    spread_pct: float = 0.0002
    borrow_rate_annual: float = 0.01

    # -- per-fill cost as a fraction of price ---------------------------
    @property
    def _market_fill_cost_pct(self) -> float:
        """Adverse cost on a market-style fill: slippage + half the spread."""
        return self.slippage_pct + self.spread_pct / 2.0

    def entry_fill_price(self, ref_price: float, direction: Direction) -> float:
        """Entry fill price after adverse slippage + half-spread."""
        cost = self._market_fill_cost_pct
        if direction == Direction.LONG:
            return ref_price * (1 + cost)
        return ref_price * (1 - cost)

    def exit_fill_price(
        self,
        ref_price: float,
        position_side: Direction,
        exit_reason: str,
    ) -> float:
        """Exit fill price.

        A "target" exit is a resting limit order and fills at the limit (no
        adverse cost). Every other exit (stop, flatten, end-of-data) is a market
        order and pays slippage + half-spread.
        """
        if exit_reason == "target":
            return ref_price
        cost = self._market_fill_cost_pct
        if position_side == Direction.LONG:
            return ref_price * (1 - cost)   # selling below the trigger
        return ref_price * (1 + cost)       # covering above the trigger

    def commission(self, quantity: int) -> float:
        """Commission for *quantity* shares (one side)."""
        return quantity * self.commission_per_share

    def borrow_cost(
        self,
        entry_price: float,
        quantity: int,
        position_side: Direction,
        days_held: float,
    ) -> float:
        """Short-borrow financing charge over the holding period (0 for longs)."""
        if position_side != Direction.SHORT or days_held <= 0:
            return 0.0
        notional = entry_price * quantity
        return notional * self.borrow_rate_annual * (days_held / 365.0)
