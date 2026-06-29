"""
backtesting.portfolio — multi-strategy, multi-symbol portfolio backtest.

Unlike the single-symbol BacktestEngine, this replays the *actual live stack*:
all registered strategies, the real RiskManager (duplicate-position filter,
correlation/concentration filter, daily-loss and drawdown kill switches), the
shared CostModel, and unified position sizing — over a merged, chronologically
ordered stream of bars across every symbol and bar size.

This is what validates live behaviour. A per-strategy backtest cannot reveal
capital contention, correlated exposure being blocked, or the portfolio-level
kill switch firing; this can.

Fill model mirrors the live engine:
  * exits are checked before strategies see the bar (stop assumed first on
    ambiguous bars),
  * realised round-trips are booked into the RiskManager so the kill switches
    engage exactly as they would live,
  * costs (commission + slippage + half-spread + short borrow) come from the
    shared CostModel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from backtesting.costs import CostModel
from backtesting.data_loader import bar_size_seconds, parse_ohlcv_csv
from backtesting.engine import BacktestTrade
from config import AppConfig
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from risk.manager import RiskManager
from strategies.base import BaseStrategy


@dataclass
class PortfolioResult:
    """Outputs of a portfolio backtest. Compatible with BacktestMetrics."""

    strategy_id: str
    symbol: str
    initial_capital: float
    final_equity: float
    start_date: str
    end_date: str
    symbols: list[str]
    equity_curve: list[tuple[datetime, float]]
    trades: list[BacktestTrade]
    per_strategy: dict[str, list[BacktestTrade]]
    signals_generated: int
    signals_taken: int
    signals_rejected: int
    rejection_reasons: dict[str, int]
    halted: bool
    halt_info: str

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.initial_capital

    def summary(
        self,
        risk_free_rate: float = 0.05,
        benchmark_returns=None,
    ) -> None:
        """Print a portfolio overview, per-strategy breakdown, significance, and
        portfolio-construction analysis (correlations + inverse-vol weights;
        plus beta/alpha vs *benchmark_returns* if provided)."""
        from backtesting.metrics import BacktestMetrics

        m = BacktestMetrics(self, self.initial_capital, risk_free_rate)
        ci_lo, ci_hi = m.sharpe_ci_95

        print("\n" + "=" * 60)
        print(f"  PORTFOLIO BACKTEST  ({self.start_date} → {self.end_date})")
        print(f"  symbols: {', '.join(self.symbols)}")
        print("=" * 60)
        print(f"  Total return    {m.total_return:+.1%}")
        print(f"  Ann. return     {m.annualised_return:+.1%}")
        print(f"  Sharpe          {m.sharpe_ratio:.2f}   "
              f"(t={m.sharpe_tstat:.2f}, 95% CI {ci_lo:.2f}…{ci_hi:.2f})")
        print(f"  PSR P[SR>0]     {m.probabilistic_sharpe_ratio:.0%}")
        print(f"  Max drawdown    {m.max_drawdown:.1%}")
        print(f"  Significant?    {'YES' if m.is_significant else 'NO — treat as noise'}")
        print("-" * 60)
        print(f"  Signals: {self.signals_generated} generated · "
              f"{self.signals_taken} taken · {self.signals_rejected} rejected")
        if self.rejection_reasons:
            for reason, n in sorted(self.rejection_reasons.items(),
                                    key=lambda kv: -kv[1]):
                print(f"      {n:4}×  {reason}")
        if self.halted:
            print(f"  ⚠  KILL SWITCH FIRED — {self.halt_info}")
        print("-" * 60)
        print(f"  {'strategy':24} {'trades':>7} {'pnl':>12} {'win%':>7}")
        for sid, trades in sorted(self.per_strategy.items()):
            closed = [t for t in trades if t.pnl is not None]
            pnl = sum(t.pnl for t in closed)
            wins = sum(1 for t in closed if t.pnl > 0)
            wr = (wins / len(closed) * 100) if closed else 0.0
            print(f"  {sid:24} {len(closed):>7} {pnl:>12,.2f} {wr:>6.1f}%")

        # Portfolio-construction analysis
        from backtesting.attribution import (
            correlation_matrix, daily_returns_from_equity_curve,
            factor_attribution, inverse_vol_weights, per_strategy_returns,
        )
        psr = per_strategy_returns(self)
        weights = inverse_vol_weights(psr)
        if weights:
            print("-" * 60)
            print("  inverse-vol weights (risk-parity proxy):")
            for sid, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                print(f"      {sid:24} {w:6.1%}")
        corr = correlation_matrix(psr)
        if len(corr) > 1:
            print("  strategy return correlations:")
            print("      " + corr.round(2).to_string().replace("\n", "\n      "))
        if benchmark_returns is not None:
            attr = factor_attribution(
                daily_returns_from_equity_curve(self.equity_curve), benchmark_returns
            )
            print("-" * 60)
            print(f"  vs benchmark:  beta={attr['beta']:.2f}  "
                  f"alpha(ann)={attr['alpha_annual']:+.1%}  "
                  f"R²={attr['r_squared']:.2f}  corr={attr['correlation']:.2f}")
        print("=" * 60 + "\n")


class PortfolioBacktestEngine:
    """Replays the full live stack over a merged multi-symbol bar stream."""

    def __init__(
        self,
        config: AppConfig,
        strategies: list[BaseStrategy],
        costs: Optional[CostModel] = None,
    ) -> None:
        self._config = config
        self._strategies = list(strategies)
        self._risk = RiskManager(config)
        self._costs = costs or CostModel()
        self._logger = logging.getLogger("backtest.portfolio")

        self._series: list[BarEvent] = []
        self._equity: float = config.risk.capital
        self._open_trades: dict[str, BacktestTrade] = {}      # symbol → trade
        self._closed_trades: list[BacktestTrade] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._last_bar: dict[str, BarEvent] = {}

        self._signals_generated = 0
        self._signals_taken = 0
        self._signals_rejected = 0
        self._rejection_reasons: dict[str, int] = {}
        self._halted = False
        self._halt_info = ""

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def add_series(self, bars: list[BarEvent]) -> None:
        """Add a pre-built list of BarEvents (one symbol/bar-size) to the mix."""
        self._series.extend(bars)

    def load_csv(self, csv_path: str | Path, symbol: str, bar_size: str) -> None:
        """Load one symbol/bar-size series from a CSV and add it to the mix."""
        self.add_series(parse_ohlcv_csv(csv_path, symbol, bar_size))

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> PortfolioResult:
        if not self._series:
            raise ValueError("No data — call add_series()/load_csv() first")

        # Merge all series; at equal timestamps, finer bars come first.
        bars = sorted(
            self._series,
            key=lambda b: (b.timestamp, bar_size_seconds(b.bar_size)),
        )
        prev_date = None

        for bar in bars:
            d = bar.timestamp.date()
            if prev_date is not None and d != prev_date:
                self._risk.reset_daily()
                for s in self._strategies:
                    if hasattr(s, "reset_daily"):
                        s.reset_daily()
            prev_date = d

            self._last_bar[bar.symbol] = bar

            # 1. Exits first (stop assumed first on ambiguous bars).
            self._check_exits(bar)

            # 2. Fan out to strategies watching this exact (symbol, bar_size).
            for strat in self._strategies:
                if (
                    strat.is_active
                    and bar.symbol in strat.symbols
                    and strat.bar_size == bar.bar_size
                ):
                    signal = strat.on_bar(bar)
                    if signal is not None:
                        self._signals_generated += 1
                        self._handle_signal(signal, strat, bar)

            # Snapshot mark-to-market equity (realised + open-position P&L) so
            # the drawdown/Sharpe reflect open-position dips, not just closes.
            self._equity_curve.append((bar.timestamp, self._mark_to_market()))

        # Close anything still open at its last seen price.
        for trade in list(self._open_trades.values()):
            last = self._last_bar[trade.symbol]
            self._close_trade(trade, float(last.close), last.timestamp, "end_of_data")
        # Final realised-equity point after everything is flat.
        self._equity_curve.append((bars[-1].timestamp, self._equity))

        symbols = sorted({b.symbol for b in bars})
        per_strategy: dict[str, list[BacktestTrade]] = {}
        for t in self._closed_trades:
            per_strategy.setdefault(t.strategy_id, []).append(t)

        return PortfolioResult(
            strategy_id="PORTFOLIO",
            symbol=",".join(symbols),
            initial_capital=self._config.risk.capital,
            final_equity=self._equity,
            start_date=bars[0].timestamp.strftime("%Y-%m-%d"),
            end_date=bars[-1].timestamp.strftime("%Y-%m-%d"),
            symbols=symbols,
            equity_curve=list(self._equity_curve),
            trades=list(self._closed_trades),
            per_strategy=per_strategy,
            signals_generated=self._signals_generated,
            signals_taken=self._signals_taken,
            signals_rejected=self._signals_rejected,
            rejection_reasons=dict(self._rejection_reasons),
            halted=self._halted,
            halt_info=self._halt_info,
        )

    # ------------------------------------------------------------------
    # Internals — mirror core.engine
    # ------------------------------------------------------------------

    def _aggregate_positions(self) -> dict[str, int]:
        combined: dict[str, int] = {}
        for s in self._strategies:
            for sym, q in s.positions.items():
                combined[sym] = combined.get(sym, 0) + q
        return combined

    def _mark_to_market(self) -> float:
        """Realised equity plus unrealised P&L of open positions at last price."""
        equity = self._equity
        for trade in self._open_trades.values():
            last = self._last_bar.get(trade.symbol)
            if last is None:
                continue
            price = float(last.close)
            if trade.direction == Direction.LONG:
                equity += (price - trade.entry_price) * trade.quantity
            else:
                equity += (trade.entry_price - price) * trade.quantity
        return equity

    def _check_exits(self, bar: BarEvent) -> None:
        trade = self._open_trades.get(bar.symbol)
        if trade is None:
            return
        hi, lo = float(bar.high), float(bar.low)
        if trade.direction == Direction.LONG:
            if lo <= trade.stop_price:
                self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")
            elif hi >= trade.target_price:
                self._close_trade(trade, trade.target_price, bar.timestamp, "target")
        else:  # SHORT
            if hi >= trade.stop_price:
                self._close_trade(trade, trade.stop_price, bar.timestamp, "stop")
            elif lo <= trade.target_price:
                self._close_trade(trade, trade.target_price, bar.timestamp, "target")

    def _handle_signal(
        self, signal: SignalEvent, strategy: BaseStrategy, bar: BarEvent
    ) -> None:
        approved, reason = self._risk.approve_signal(signal, self._aggregate_positions())
        if not approved:
            self._signals_rejected += 1
            key = reason.split(":")[0]
            self._rejection_reasons[key] = self._rejection_reasons.get(key, 0) + 1
            return

        if signal.direction == Direction.FLAT:
            trade = self._open_trades.get(signal.symbol)
            if trade is not None:
                self._close_trade(trade, float(bar.close), bar.timestamp, "flatten")
            return

        if signal.symbol in self._open_trades:
            return  # duplicate (approve_signal normally blocks this already)

        levels = strategy.get_pending_levels(signal.symbol)
        if levels is None:
            return

        qty = self._risk.size_position(signal, levels["entry"], levels["stop"])
        if qty == 0:
            return

        fill_price = self._costs.entry_fill_price(levels["entry"], signal.direction)
        entry_commission = self._costs.commission(qty)
        self._equity -= entry_commission

        trade = BacktestTrade(
            strategy_id=strategy.strategy_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_time=bar.timestamp,
            exit_time=None,
            entry_price=fill_price,
            exit_price=None,
            quantity=qty,
            pnl=None,
            commission=entry_commission,   # holds entry commission until close
            exit_reason="",
            stop_price=levels["stop"],
            target_price=levels["target"],
        )
        self._open_trades[signal.symbol] = trade

        # Notify the owning strategy (execution side == position side on entry).
        self._notify_fill(strategy.strategy_id, FillEvent(
            strategy_id=strategy.strategy_id, symbol=signal.symbol,
            timestamp=bar.timestamp, direction=signal.direction, quantity=qty,
            fill_price=Decimal(str(fill_price)),
            commission=Decimal(str(entry_commission)),
        ))
        self._signals_taken += 1

    def _close_trade(
        self, trade: BacktestTrade, raw_exit_price: float,
        exit_time: datetime, exit_reason: str,
    ) -> None:
        symbol = trade.symbol
        qty = trade.quantity
        exit_price = self._costs.exit_fill_price(raw_exit_price, trade.direction, exit_reason)

        if trade.direction == Direction.LONG:
            gross = (exit_price - trade.entry_price) * qty
        else:
            gross = (trade.entry_price - exit_price) * qty

        days_held = max((exit_time - trade.entry_time).total_seconds() / 86_400.0, 0.0)
        borrow = self._costs.borrow_cost(trade.entry_price, qty, trade.direction, days_held)
        exit_commission = self._costs.commission(qty)
        entry_commission = trade.commission
        net_pnl = gross - exit_commission - borrow
        self._equity += net_pnl

        # Book the full round-trip into the RiskManager so the daily-loss and
        # drawdown kill switches engage exactly as they do live.
        was_halted = self._risk.is_halted
        self._risk.record_close(
            symbol, trade.entry_price, exit_price, qty, trade.direction,
            commission=entry_commission + exit_commission + borrow,
        )
        if self._risk.is_halted and not was_halted:
            self._halted = True
            self._halt_info = (
                f"{exit_time.date()} — daily_pnl={self._risk.daily_pnl:.2f}, "
                f"drawdown={self._risk.current_drawdown_pct:.1%}"
            )

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = net_pnl
        trade.commission = entry_commission + exit_commission
        trade.borrow_cost = borrow
        trade.exit_reason = exit_reason
        self._closed_trades.append(trade)
        del self._open_trades[symbol]

        # Notify the owning strategy with the closing execution side.
        close_dir = Direction.SHORT if trade.direction == Direction.LONG else Direction.LONG
        self._notify_fill(trade.strategy_id, FillEvent(
            strategy_id=trade.strategy_id, symbol=symbol, timestamp=exit_time,
            direction=close_dir, quantity=qty,
            fill_price=Decimal(str(exit_price)),
            commission=Decimal(str(exit_commission)),
        ))

    def _notify_fill(self, strategy_id: str, fill: FillEvent) -> None:
        for s in self._strategies:
            if s.strategy_id == strategy_id:
                s.on_fill(fill)
