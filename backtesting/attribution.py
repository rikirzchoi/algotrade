"""
backtesting.attribution — performance attribution & portfolio construction.

Answers the questions a quant allocator asks:
  * "How much of this return is just market beta you could buy for free?"
    → factor_attribution() regresses strategy returns on a benchmark and
      reports alpha (annualised), beta, R², and correlation.
  * "Are these strategies actually diversifying, or the same bet?"
    → correlation_matrix() across per-strategy return streams.
  * "How should capital be split to balance risk?"
    → inverse_vol_weights() (a simple risk-parity proxy).

All functions take pandas return Series/dicts; helpers derive those from a
PortfolioResult or an equity curve.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return-series builders
# ---------------------------------------------------------------------------

def daily_returns_from_equity_curve(
    curve: list[tuple[datetime, float]],
) -> pd.Series:
    """Daily simple returns from an equity curve (last value per calendar day)."""
    if not curve:
        return pd.Series(dtype=float)
    s = pd.Series({ts: eq for ts, eq in curve}, dtype=float)
    s.index = pd.to_datetime(s.index)
    daily = s.groupby(s.index.date).last()
    return daily.pct_change().dropna()


def per_strategy_returns(result, capital: Optional[float] = None) -> dict[str, pd.Series]:
    """Per-strategy daily return series derived from closed trades.

    Return for a day = (sum of that strategy's realised P&L that day) / capital.
    Days with no closed trade are 0 (the strategy was flat / in cash).
    """
    cap = capital if capital is not None else getattr(result, "initial_capital", 1.0)
    out: dict[str, pd.Series] = {}
    for sid, trades in result.per_strategy.items():
        rows = [
            (t.exit_time.date(), t.pnl)
            for t in trades
            if t.exit_time is not None and t.pnl is not None
        ]
        if not rows:
            out[sid] = pd.Series(dtype=float)
            continue
        df = pd.DataFrame(rows, columns=["date", "pnl"])
        daily = df.groupby("date")["pnl"].sum() / cap
        daily.index = pd.to_datetime(daily.index)
        out[sid] = daily.sort_index()
    return out


# ---------------------------------------------------------------------------
# Factor attribution
# ---------------------------------------------------------------------------

def factor_attribution(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float]:
    """Regress *returns* on *benchmark_returns*: alpha/beta/R²/correlation.

    alpha is annualised (daily intercept × 252). Returns zeros if there is
    insufficient overlapping data.
    """
    def _norm(s: pd.Series) -> pd.Series:
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        return pd.Series(s.to_numpy(dtype=float), index=idx.normalize())

    joined = pd.concat(
        [_norm(returns).rename("r"), _norm(benchmark_returns).rename("b")], axis=1
    ).dropna()
    n = len(joined)
    if n < 3:
        return {"alpha_annual": 0.0, "beta": 0.0, "r_squared": 0.0,
                "correlation": 0.0, "n_obs": float(n)}

    from scipy.stats import linregress
    res = linregress(joined["b"].to_numpy(), joined["r"].to_numpy())
    return {
        "alpha_annual": float(res.intercept * _TRADING_DAYS),
        "beta": float(res.slope),
        "r_squared": float(res.rvalue ** 2),
        "correlation": float(res.rvalue),
        "n_obs": float(n),
    }


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def correlation_matrix(per_strategy: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation matrix across per-strategy daily return streams.

    Series are aligned on the union of dates; missing days are treated as 0
    (flat) so a strategy that rarely trades is not dropped.
    """
    streams = {k: v for k, v in per_strategy.items() if not v.empty}
    if not streams:
        return pd.DataFrame()
    df = pd.DataFrame(streams).fillna(0.0)
    return df.corr()


def inverse_vol_weights(per_strategy: dict[str, pd.Series]) -> dict[str, float]:
    """Inverse-volatility (risk-parity proxy) capital weights, summing to 1.

    Strategies with higher return volatility receive proportionally less
    capital. Zero-vol or empty strategies are excluded.
    """
    inv: dict[str, float] = {}
    for sid, series in per_strategy.items():
        if series.empty:
            continue
        vol = float(series.std())
        if vol > 0:
            inv[sid] = 1.0 / vol
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {sid: w / total for sid, w in inv.items()}


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised volatility of a daily return series."""
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(_TRADING_DAYS))
