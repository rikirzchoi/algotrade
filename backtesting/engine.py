"""
backtesting.engine — historical simulation that reuses live strategy classes.

The BacktestEngine reads stored OHLCV bars from a CSV file, wraps each row in
a BarEvent, and dispatches it through the same strategy.on_bar() pipeline used
in live trading.  This ensures backtested results reflect the same logic and
risk rules that run in production.

Fill simulation (deliberately conservative)
-------------------------------------------
- Entries fill at the signal's entry price plus slippage (default 0.05 %).
- When a single bar's range spans BOTH the stop and the target, the stop is
  assumed to hit first (worst case) — no optimistic "target first" bias.
- Stop / flatten / end-of-data exits are market-style and take adverse
  slippage; target exits are resting limit orders and fill at the limit.
- Position sizing uses the SAME path as live (RiskManager.size_position:
  fixed-fractional core + macro overlay), so sizes match production.
- Commission is a flat per-share amount (default $0.005, IBKR rate).

Limitations
-----------
- No partial fills
- No intraday liquidity / market-impact modelling
- No short-borrow cost
- Macro overlay uses the CURRENT regime across all history (no historical series)
- Single symbol per engine instance (portfolio backtest not yet supported)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

from config import AppConfig
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from backtesting.costs import CostModel
from backtesting.data_loader import parse_ohlcv_csv
from risk.manager import RiskManager
from strategies.base import BaseStrategy

_ET = pytz.timezone("America/New_York")


@dataclass
class BacktestTrade:
    """Record of one completed (or in-progress) round-trip trade."""

    strategy_id: str
    symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    commission: float
    exit_reason: str          # "target" | "stop" | "flatten" | "end_of_data"
    stop_price: float = 0.0   # used by engine to check bar lows/highs
    target_price: float = 0.0
    borrow_cost: float = 0.0  # short-borrow financing charge (shorts only)


@dataclass
class BacktestResult:
    """All outputs produced by a completed backtest run."""

    strategy_id: str
    symbol: str
    start_date: str
    end_date: str
    trades: list[BacktestTrade]
    equity_curve: list[tuple[datetime, float]]
    total_bars: int
    signals_generated: int
    signals_taken: int


class BacktestEngine:
    """
    Replays historical bars through a live strategy instance.

    Usage::

        cfg = AppConfig()
        strategy = MomentumBreakoutStrategy(cfg)
        engine = BacktestEngine(cfg, strategy)
        engine.load_csv("data/SPY.csv")
        result = engine.run()
    """

    def __init__(self, config: AppConfig, strategy: BaseStrategy) -> None:
        self._config = config
        self._strategy = strategy
        self._bars: list[BarEvent] = []
        self._logger = logging.getLogger("backtest.engine")

        # Same sizing path as live (fixed-fractional core + macro overlay), so
        # backtested position sizes match what the live engine would trade.
        self._risk = RiskManager(config)

        # Simulated account state
        self._equity: float = config.risk.capital
        self._open_trades: dict[str, BacktestTrade] = {}   # symbol → open trade
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._signals_generated: int = 0
        self._signals_taken: int = 0

        # Shared, conservative cost model (also used by the portfolio engine).
        self._costs = CostModel()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_csv(self, csv_path: str | Path, symbol: Optional[str] = None) -> None:
        """Load historical OHLCV data from *csv_path* (see data_loader)."""
        self._bars = parse_ohlcv_csv(csv_path, symbol)
        self._logger.info(
            "Loaded %d bars for %s from %s to %s",
            len(self._bars),
            self._bars[0].symbol,
            self._bars[0].timestamp,
            self._bars[-1].timestamp,
        )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Iterate through loaded bars in chronological order, simulate fills,
        and return a BacktestResult.
        """
        if not self._bars:
            raise ValueError("No bars loaded — call load_csv() first")

        prev_date = None

        for bar in self._bars:
            symbol = bar.symbol
            bar_date = bar.timestamp.date()

            # -- Daily reset (new trading day) ---------------------------
            if prev_date is not None and bar_date != prev_date:
                if hasattr(self._strategy, "reset_daily"):
                    self._strategy.reset_daily()
            prev_date = bar_date

            # -- Step 1: check open trade exits BEFORE calling strategy --
            if symbol in self._open_trades:
                trade = self._open_trades[symbol]
                bar_high = float(bar.high)
                bar_low = float(bar.low)

                # Stop is checked BEFORE target: when a single bar's range spans
                # both levels we cannot know the intrabar path, so we assume the
                # worse outcome (stop hit first). This avoids the optimistic bias
                # of booking the win whenever a bar straddles both.
                if trade.direction == Direction.LONG:
                    if bar_low <= trade.stop_price:
                        self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")
                    elif bar_high >= trade.target_price:
                        self._close_trade(trade, trade.target_price, bar.timestamp, "target")
                elif trade.direction == Direction.SHORT:
                    if bar_high >= trade.stop_price:
                        self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")
                    elif bar_low <= trade.target_price:
                        self._close_trade(trade, trade.target_price, bar.timestamp, "target")

            # -- Step 2: call strategy and snapshot equity ---------------
            signal = self._strategy.on_bar(bar)
            # Mark-to-market: realised equity + unrealised P&L of any open
            # position at this bar's close, so drawdown sees open-position dips.
            mtm = self._equity
            open_trade = self._open_trades.get(symbol)
            if open_trade is not None:
                px = float(bar.close)
                if open_trade.direction == Direction.LONG:
                    mtm += (px - open_trade.entry_price) * open_trade.quantity
                else:
                    mtm += (open_trade.entry_price - px) * open_trade.quantity
            self._equity_curve.append((bar.timestamp, mtm))

            # -- Step 3: process the signal ------------------------------
            if signal is None:
                continue

            self._signals_generated += 1
            close_price = float(bar.close)

            if signal.direction == Direction.FLAT:
                if symbol in self._open_trades:
                    self._close_trade(
                        self._open_trades[symbol], close_price, bar.timestamp, "flatten"
                    )
                continue

            # LONG or SHORT — skip if already in a position
            if symbol in self._open_trades:
                continue

            levels: Optional[dict] = None
            if hasattr(self._strategy, "get_pending_levels"):
                levels = self._strategy.get_pending_levels(symbol)

            if levels is None:
                continue

            qty = self._risk.size_position(signal, levels["entry"], levels["stop"])
            if qty == 0:
                continue

            fill_price = self._costs.entry_fill_price(levels["entry"], signal.direction)

            entry_commission = self._costs.commission(qty)
            self._equity -= entry_commission

            trade = BacktestTrade(
                strategy_id=self._strategy.strategy_id,
                symbol=symbol,
                direction=signal.direction,
                entry_time=bar.timestamp,
                exit_time=None,
                entry_price=fill_price,
                exit_price=None,
                quantity=qty,
                pnl=None,
                commission=entry_commission,
                exit_reason="",
                stop_price=levels["stop"],
                target_price=levels["target"],
            )
            self._open_trades[symbol] = trade

            fill = FillEvent(
                strategy_id=self._strategy.strategy_id,
                symbol=symbol,
                timestamp=bar.timestamp,
                direction=signal.direction,
                quantity=qty,
                fill_price=Decimal(str(fill_price)),
                commission=Decimal(str(entry_commission)),
                client_order_id=None,
                ib_exec_id="",
            )
            # on_fill handles update_position internally for LONG/SHORT
            self._strategy.on_fill(fill)
            self._signals_taken += 1

        # -- Step 5: close remaining open trades at end of data ----------
        if self._bars:
            last_bar = self._bars[-1]
            for trade in list(self._open_trades.values()):
                self._close_trade(
                    trade, float(last_bar.close), last_bar.timestamp, "end_of_data"
                )

        # -- Step 6: build result ----------------------------------------
        bars = self._bars
        symbol = bars[0].symbol if bars else ""

        return BacktestResult(
            strategy_id=self._strategy.strategy_id,
            symbol=symbol,
            start_date=bars[0].timestamp.strftime("%Y-%m-%d") if bars else "",
            end_date=bars[-1].timestamp.strftime("%Y-%m-%d") if bars else "",
            trades=list(self._closed_trades),
            equity_curve=list(self._equity_curve),
            total_bars=len(self._bars),
            signals_generated=self._signals_generated,
            signals_taken=self._signals_taken,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> None:
        """Close *trade* at *exit_price*, update equity, and notify strategy."""
        symbol = trade.symbol
        qty = trade.quantity

        # Apply the shared cost model: market exits pay slippage + half-spread;
        # limit "target" exits fill clean.
        exit_price = self._costs.exit_fill_price(exit_price, trade.direction, exit_reason)

        if trade.direction == Direction.LONG:
            gross_pnl = (exit_price - trade.entry_price) * qty
        else:
            gross_pnl = (trade.entry_price - exit_price) * qty

        # Short positions accrue a borrow/financing charge for the holding period.
        days_held = max((exit_time - trade.entry_time).total_seconds() / 86_400.0, 0.0)
        borrow_cost = self._costs.borrow_cost(
            trade.entry_price, qty, trade.direction, days_held
        )

        exit_commission = self._costs.commission(qty)
        net_pnl = gross_pnl - exit_commission - borrow_cost
        self._equity += net_pnl

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.commission = trade.commission + exit_commission
        trade.borrow_cost = borrow_cost
        trade.exit_reason = exit_reason

        self._closed_trades.append(trade)
        del self._open_trades[symbol]

        # Clear strategy position tracking directly (spec step 1)
        qty_delta = -qty if trade.direction == Direction.LONG else qty
        self._strategy.update_position(symbol, qty_delta)

        self._logger.debug(
            "Closed %s %s qty=%d @ %.2f | pnl=%.2f | reason=%s",
            trade.direction.value, symbol, qty, exit_price, net_pnl, exit_reason,
        )
