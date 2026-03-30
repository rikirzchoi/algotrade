"""
backtesting.engine — historical simulation that reuses live strategy classes.

The BacktestEngine reads stored OHLCV bars from a CSV file, wraps each row in
a BarEvent, and dispatches it through the same strategy.on_bar() pipeline used
in live trading.  This ensures backtested results reflect the same logic and
risk rules that run in production.

Fill simulation
---------------
Entries receive the signal's pending-levels entry price with a configurable
slippage (default: 0.05 %).  Exits at target/stop are assumed to fill at the
exact level (no additional slippage — conservative assumption).
Commission is modelled as a flat per-share amount (default: $0.005 IBKR rate).

Limitations
-----------
- No partial fills
- No intraday liquidity modelling
- Assumes fills occur at the simulated price with 100 % certainty
- Single symbol per engine instance (multi-symbol not yet supported)
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
from risk.sizing import fixed_fractional
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

        # Simulated account state
        self._equity: float = config.risk.capital
        self._open_trades: dict[str, BacktestTrade] = {}   # symbol → open trade
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._signals_generated: int = 0
        self._signals_taken: int = 0

        # Commission / slippage model
        self._commission_per_share: float = 0.005   # IBKR typical
        self._slippage_pct: float = 0.0005          # 0.05 %

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_csv(self, csv_path: str | Path, symbol: Optional[str] = None) -> None:
        """
        Load historical OHLCV data from *csv_path*.

        Accepts both lowercase (timestamp/open/…) and Yahoo Finance-style
        (Date/Open/…) column names, as well as the yfinance multi-level header
        format where the first two rows are field names and ticker names::

            Price,Close,High,Low,Open,Volume
            Ticker,SPY,SPY,SPY,SPY,SPY
            2024-01-02,...

        Date-only timestamps are assigned a close-of-day time of 16:00:00 ET.
        Bar size is inferred from the median interval between consecutive bars.
        """
        csv_path = Path(csv_path)

        if symbol is None:
            symbol = csv_path.stem.upper()

        # Peek at the first few rows to detect format and retain for error reporting
        raw_peek = pd.read_csv(csv_path, header=None, nrows=5)

        # Detect yfinance multi-level format: the second row (index 1) contains
        # non-numeric ticker symbols rather than data values.
        is_yfinance = False
        if len(raw_peek) >= 2:
            second_row_val = str(raw_peek.iloc[1, 1])
            try:
                float(second_row_val)
            except ValueError:
                is_yfinance = True

        if is_yfinance:
            # Row 0 supplies field names (Price/Close/High/Low/Open/Volume);
            # row 1 supplies ticker names — both rows are skipped as data.
            col_names = list(raw_peek.iloc[0])
            df = pd.read_csv(csv_path, skiprows=2, names=col_names)
            # First column (labelled "Price") holds dates → rename to "timestamp"
            rename: dict[str, str] = {col_names[0]: "timestamp"}
            for col in col_names[1:]:
                rename[col] = col.lower()
            df = df.rename(columns=rename)
        else:
            df = pd.read_csv(csv_path)
            # Normalise column names to lowercase variants
            col_lower = {c.lower(): c for c in df.columns}
            rename = {}
            for target, candidates in [
                ("timestamp", ["timestamp", "date", "datetime", "time"]),
                ("open",      ["open"]),
                ("high",      ["high"]),
                ("low",       ["low"]),
                ("close",     ["close", "adj close"]),
                ("volume",    ["volume"]),
            ]:
                for cand in candidates:
                    if cand in col_lower:
                        rename[col_lower[cand]] = target
                        break
            df = df.rename(columns=rename)

        raw: list[tuple[datetime, float, float, float, float, int]] = []
        for _, row in df.iterrows():
            try:
                ts = pd.to_datetime(str(row["timestamp"]))
            except Exception:
                continue

            # Date-only → close of day 16:00 ET
            if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
                ts = ts.replace(hour=16, minute=0, second=0)

            if ts.tzinfo is None:
                ts = _ET.localize(ts)
            else:
                ts = ts.astimezone(_ET)

            raw.append((
                ts.to_pydatetime(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(float(row.get("volume", 0))),
            ))

        raw.sort(key=lambda x: x[0])

        # Infer bar size from median interval between consecutive rows
        bar_size = "1 day"
        if len(raw) >= 2:
            deltas = [
                (raw[i + 1][0] - raw[i][0]).total_seconds()
                for i in range(min(len(raw) - 1, 20))
            ]
            median_secs = sorted(deltas)[len(deltas) // 2]
            if median_secs <= 120:
                bar_size = "1 min"
            elif median_secs <= 1_000:
                bar_size = "15 mins"
            else:
                bar_size = "1 day"

        self._bars = [
            BarEvent(
                symbol=symbol,
                timestamp=r[0],
                open=Decimal(str(r[1])),
                high=Decimal(str(r[2])),
                low=Decimal(str(r[3])),
                close=Decimal(str(r[4])),
                volume=r[5],
                bar_size=bar_size,
                vwap=Decimal(str(r[4])),   # close as VWAP approximation
            )
            for r in raw
        ]

        if not self._bars:
            preview = raw_peek.to_string(index=False)
            raise ValueError(
                f"load_csv loaded 0 bars from {csv_path}. "
                f"Check that the file contains valid OHLCV data.\n"
                f"First few raw rows:\n{preview}"
            )

        self._logger.info(
            "Loaded %d bars for %s from %s to %s",
            len(self._bars),
            symbol,
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

                if trade.direction == Direction.LONG:
                    if bar_high >= trade.target_price:
                        self._close_trade(trade, trade.target_price, bar.timestamp, "target")
                    elif bar_low <= trade.stop_price:
                        self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")
                elif trade.direction == Direction.SHORT:
                    if bar_low <= trade.target_price:
                        self._close_trade(trade, trade.target_price, bar.timestamp, "target")
                    elif bar_high >= trade.stop_price:
                        self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")

            # -- Step 2: call strategy and snapshot equity ---------------
            signal = self._strategy.on_bar(bar)
            self._equity_curve.append((bar.timestamp, self._equity))

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

            qty = fixed_fractional(
                self._equity,
                self._config.risk.risk_per_trade_pct,
                levels["entry"],
                levels["stop"],
                self._config.risk.max_position_size,
            )
            if qty == 0:
                continue

            if signal.direction == Direction.LONG:
                fill_price = levels["entry"] * (1 + self._slippage_pct)
            else:
                fill_price = levels["entry"] * (1 - self._slippage_pct)

            entry_commission = qty * self._commission_per_share
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

        if trade.direction == Direction.LONG:
            gross_pnl = (exit_price - trade.entry_price) * qty
        else:
            gross_pnl = (trade.entry_price - exit_price) * qty

        exit_commission = qty * self._commission_per_share
        net_pnl = gross_pnl - exit_commission
        self._equity += net_pnl

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.commission = trade.commission + exit_commission
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
