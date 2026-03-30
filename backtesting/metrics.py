"""
backtesting.metrics — standardised performance statistics for backtest results.

All metrics are computed from a BacktestResult produced by BacktestEngine.
The BacktestMetrics class exposes every metric as a read-only property so
callers can access only what they need without triggering unnecessary work.

Metrics
-------
- Total return
- Annualised return (CAGR)
- Sharpe ratio (annualised, configurable risk-free rate)
- Maximum drawdown (fraction) and duration (calendar days)
- Win rate, avg win, avg loss, expectancy, profit factor
- Trade count, trades per day
- Best / worst trade accessors
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """
    Performance analytics computed from a BacktestResult.

    Parameters
    ----------
    result          : BacktestResult returned by BacktestEngine.run()
    initial_capital : starting capital used in the backtest
    risk_free_rate  : annualised risk-free rate for Sharpe (default 5 %)
    """

    def __init__(
        self,
        result: object,          # BacktestResult — typed as object to avoid circular import
        initial_capital: float,
        risk_free_rate: float = 0.05,
    ) -> None:
        self._result = result
        self._initial_capital = initial_capital
        self._risk_free_rate = risk_free_rate

        # Pre-compute daily returns from the equity curve
        if result.equity_curve:
            series = pd.Series(
                {ts: eq for ts, eq in result.equity_curve},
                dtype=float,
            )
            series.index = pd.to_datetime(series.index)
            # Last equity value per calendar day
            daily = series.groupby(series.index.date).last()
            self._daily_equity: pd.Series = daily.astype(float)
            self._daily_returns: pd.Series = daily.pct_change().dropna()
        else:
            self._daily_equity = pd.Series(dtype=float)
            self._daily_returns = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    @property
    def _final_equity(self) -> float:
        if self._daily_equity.empty:
            return self._initial_capital
        return float(self._daily_equity.iloc[-1])

    @property
    def _trading_days(self) -> int:
        return max(len(self._daily_returns), 1)

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------

    @property
    def total_return(self) -> float:
        """Total return as a fraction (e.g. 0.12 = 12 %)."""
        return (self._final_equity - self._initial_capital) / self._initial_capital

    @property
    def annualised_return(self) -> float:
        """CAGR implied by the equity curve."""
        if self._trading_days < 2:
            return 0.0
        try:
            return (self._final_equity / self._initial_capital) ** (252 / self._trading_days) - 1
        except (ZeroDivisionError, ValueError):
            return 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio.  Returns 0.0 if std is zero."""
        if self._daily_returns.empty:
            return 0.0
        std = float(self._daily_returns.std())
        if std == 0.0:
            return 0.0
        mean = float(self._daily_returns.mean())
        return (mean - self._risk_free_rate / 252) / std * math.sqrt(252)

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    @property
    def max_drawdown(self) -> float:
        """Peak-to-trough maximum drawdown as a positive fraction (e.g. 0.15 = 15 %)."""
        if self._daily_equity.empty:
            return 0.0
        rolling_peak = self._daily_equity.cummax()
        drawdowns = (self._daily_equity - rolling_peak) / rolling_peak
        return float(-drawdowns.min())

    @property
    def max_drawdown_duration(self) -> int:
        """Number of calendar days in the longest drawdown period."""
        if len(self._daily_equity) < 2:
            return 0

        peak_value = float(self._daily_equity.iloc[0])
        peak_date = self._daily_equity.index[0]
        max_duration = 0

        for date, eq in self._daily_equity.items():
            if eq >= peak_value:
                peak_value = float(eq)
                peak_date = date
            else:
                duration = (date - peak_date).days
                max_duration = max(max_duration, duration)

        return max_duration

    # ------------------------------------------------------------------
    # Trade metrics
    # ------------------------------------------------------------------

    @property
    def winning_trades(self):
        """List of BacktestTrade instances with pnl > 0."""
        return [t for t in self._result.trades if t.pnl is not None and t.pnl > 0]

    @property
    def losing_trades(self):
        """List of BacktestTrade instances with pnl < 0."""
        return [t for t in self._result.trades if t.pnl is not None and t.pnl < 0]

    @property
    def trade_count(self) -> int:
        return len(self._result.trades)

    @property
    def trades_per_day(self) -> float:
        return self.trade_count / self._trading_days

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades with pnl > 0."""
        if self.trade_count == 0:
            return 0.0
        return len(self.winning_trades) / self.trade_count

    @property
    def avg_win(self) -> float:
        """Mean pnl of winning trades."""
        wins = self.winning_trades
        if not wins:
            return 0.0
        return sum(t.pnl for t in wins) / len(wins)   # type: ignore[arg-type]

    @property
    def avg_loss(self) -> float:
        """Mean pnl of losing trades (returned as a negative float)."""
        losses = self.losing_trades
        if not losses:
            return 0.0
        return sum(t.pnl for t in losses) / len(losses)  # type: ignore[arg-type]

    @property
    def expectancy(self) -> float:
        """Expected dollar profit per trade."""
        return self.avg_win * self.win_rate + self.avg_loss * (1 - self.win_rate)

    @property
    def profit_factor(self) -> float:
        """Gross profit / |gross loss|.  Returns 0.0 if no losing trades."""
        gross_profit = sum(t.pnl for t in self.winning_trades)   # type: ignore[arg-type]
        gross_loss = abs(sum(t.pnl for t in self.losing_trades))  # type: ignore[arg-type]
        if gross_loss == 0:
            return 0.0
        return gross_profit / gross_loss

    @property
    def best_trade(self):
        """BacktestTrade with the highest pnl, or None."""
        trades = [t for t in self._result.trades if t.pnl is not None]
        if not trades:
            return None
        return max(trades, key=lambda t: t.pnl)

    @property
    def worst_trade(self):
        """BacktestTrade with the lowest pnl, or None."""
        trades = [t for t in self._result.trades if t.pnl is not None]
        if not trades:
            return None
        return min(trades, key=lambda t: t.pnl)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted performance summary table to stdout."""
        r = self._result
        width = 43   # inner width between the │ borders

        def row(label: str, value: str) -> str:
            content = f"  {label:<22}{value}"
            return f"│{content:<{width}}│"

        divider = f"├{'─' * width}┤"
        top     = f"┌{'─' * width}┐"
        bottom  = f"└{'─' * width}┘"

        title1 = f"  Backtest: {r.strategy_id} / {r.symbol}"
        title2 = f"  {r.start_date} → {r.end_date}"

        lines = [
            top,
            f"│{title1:<{width}}│",
            f"│{title2:<{width}}│",
            divider,
            row("Total return",    f"{self.total_return:.1%}"),
            row("Ann. return",     f"{self.annualised_return:.1%}"),
            row("Sharpe ratio",    f"{self.sharpe_ratio:.2f}"),
            row("Max drawdown",    f"{self.max_drawdown:.1%}"),
            row("Max DD duration", f"{self.max_drawdown_duration} days"),
            divider,
            row("Trades",          str(self.trade_count)),
            row("Trades/day",      f"{self.trades_per_day:.2f}"),
            row("Win rate",        f"{self.win_rate:.1%}"),
            row("Avg win",         f"${self.avg_win:.2f}"),
            row("Avg loss",        f"${self.avg_loss:.2f}"),
            row("Expectancy",      f"${self.expectancy:.2f}"),
            row("Profit factor",   f"{self.profit_factor:.2f}"),
            bottom,
        ]
        print("\n".join(lines))

    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot the equity curve with drawdown periods shaded.

        If *save_path* is provided the figure is saved to disk instead of
        being shown interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("matplotlib not available — cannot plot equity curve")
            return

        if self._daily_equity.empty:
            logger.warning("No equity data to plot")
            return

        r = self._result
        dates = [pd.Timestamp(d) for d in self._daily_equity.index]
        equity = self._daily_equity.values

        fig, ax = plt.subplots(figsize=(12, 5))

        # Equity curve
        ax.plot(dates, equity, color="steelblue", linewidth=1.5, label="Equity")

        # Initial capital baseline
        ax.axhline(
            self._initial_capital,
            color="gray",
            linewidth=0.8,
            linestyle="--",
            label=f"Initial capital (${self._initial_capital:,.0f})",
        )

        # Shade drawdown periods
        rolling_peak = self._daily_equity.cummax().values
        in_dd = False
        dd_start: Optional[pd.Timestamp] = None
        for i, (d, eq, peak) in enumerate(zip(dates, equity, rolling_peak)):
            currently_down = eq < peak
            if currently_down and not in_dd:
                in_dd = True
                dd_start = d
            elif not currently_down and in_dd:
                in_dd = False
                ax.axvspan(dd_start, d, color="salmon", alpha=0.3)
        if in_dd and dd_start is not None:
            ax.axvspan(dd_start, dates[-1], color="salmon", alpha=0.3)

        ax.set_title(f"Equity Curve — {r.strategy_id} / {r.symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.legend()
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            self._logger_ref().info("Equity curve saved to %s", save_path)
        else:
            plt.show()

        plt.close(fig)

    def _logger_ref(self):
        return logger
