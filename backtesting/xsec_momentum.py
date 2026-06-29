"""
xsec_momentum — Gate-1 research backtest for CROSS-SECTIONAL momentum (relative
strength) across a homogeneous universe of US sector ETFs.

Thesis (distinct from time-series trend): within a peer group, assets that have
*outperformed their peers* over the past ~12 months keep outperforming over the
next month (Jegadeesh-Titman 1993; Moskowitz-Grinblatt industry momentum 1999).
The edge is RELATIVE — rank assets against each other and go long winners / short
losers — whereas trend_research is ABSOLUTE (each asset vs its own zero line).
Counterparty: slow institutional reallocation + underreaction to sector-level news.
Why this fits our infra: daily data, monthly turnover, no latency, and sector ETFs
do not delist → survivorship-clean on free yfinance data.

PRE-COMMITTED DESIGN (fixed before seeing results, per TRADING_RULES.md §10):
  * universe        = 11 SPDR sector ETFs (homogeneous, ranking is meaningful)
  * signal          = trailing 12-1 month return (skip most recent month to dodge
                      the well-known 1-month reversal) — the classic momentum window
  * portfolio       = long top tercile, short bottom tercile, equal-weight per leg,
                      dollar-neutral, gross exposure normalised to 1 (unlevered)
  * rebalance       = month-end; weights lagged 1 day (no same-day look-ahead)
  * costs           = 10 bps per unit of weight traded (same as trend_research)
  * start           = once ≥6 markets are live (need 2 names per tercile)
NO parameter sweeps. One configuration, one verdict. Also reports correlation to
the validated trend strategy — the whole point is an *uncorrelated* second edge.

Run:  .venv/bin/python -m backtesting.xsec_momentum
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting.metrics import BacktestMetrics

CAPITAL = 100_000.0
COST_PER_TURNOVER = 0.0010          # 10 bps per unit of weight traded
MOM_MONTHS = 12                     # trailing-return lookback
SKIP_MONTHS = 1                     # skip most recent month (12-1 momentum)

# Homogeneous universe: the 11 SPDR sector ETFs. XLRE (2015) and XLC (2018) join
# late; the breadth filter handles the early years when only 9 are live.
UNIVERSE = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health care", "XLI": "Industrials", "XLY": "Cons. discretionary",
    "XLP": "Cons. staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real estate", "XLC": "Communications",
}


def _metrics(equity: pd.Series, label: str) -> BacktestMetrics:
    res = SimpleNamespace(
        strategy_id=label, symbol="sectors",
        start_date=str(equity.index[0].date()), end_date=str(equity.index[-1].date()),
        equity_curve=list(zip(equity.index.to_pydatetime(), equity.to_numpy(float))),
        trades=[],
    )
    return BacktestMetrics(res, CAPITAL, risk_free_rate=0.0)


def _stats(r: pd.Series) -> tuple[float, float, float]:
    """Annual return, annualised Sharpe (rf=0), and max drawdown for a return series."""
    r = r.dropna()
    if r.empty or r.std() == 0:
        return 0.0, 0.0, 0.0
    eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (252 / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252)
    max_dd = -(eq / eq.cummax() - 1).min()
    return ann, sharpe, max_dd


def _report(equity: pd.Series, label: str) -> None:
    m = _metrics(equity, label)
    lo, hi = m.sharpe_ci_95
    ann = (equity.iloc[-1] / equity.iloc[0]) ** (252 / max(len(equity), 1)) - 1
    print(f"\n  {label}")
    print(f"    period        {equity.index[0].date()} → {equity.index[-1].date()}")
    print(f"    total return  {(equity.iloc[-1]/equity.iloc[0]-1):+.1%}")
    print(f"    ann. return   {ann:+.1%}")
    print(f"    Sharpe        {m.sharpe_ratio:.2f}  (t={m.sharpe_tstat:.1f}, "
          f"95% CI {lo:.2f}…{hi:.2f})")
    print(f"    PSR P[SR>0]   {m.probabilistic_sharpe_ratio:.0%}")
    print(f"    max drawdown  {m.max_drawdown:.1%}")
    print(f"    significant?  {'YES' if m.is_significant else 'no (by trade count rule)'}")


def fetch_universe(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted daily closes for the universe (max history)."""
    raw = yf.download(tickers, period="max", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    return raw.dropna(how="all").sort_index()


def _tercile_weights(mom_row: pd.Series) -> pd.Series:
    """Long top tercile / short bottom tercile, dollar-neutral, gross abs = 1."""
    w = pd.Series(0.0, index=mom_row.index)
    valid = mom_row.dropna()
    n = len(valid)
    if n < 6:                       # need ≥2 names per tercile to diversify the leg
        return w
    k = n // 3
    ranked = valid.sort_values()
    longs = ranked.index[-k:]
    shorts = ranked.index[:k]
    w[longs] = 0.5 / k              # long leg sums to +0.5
    w[shorts] = -0.5 / k            # short leg sums to -0.5  → gross = 1
    return w


def xsec_returns(px: pd.DataFrame,
                 mom_months: int = MOM_MONTHS,
                 skip_months: int = SKIP_MONTHS) -> pd.Series:
    """Daily net returns of the cross-sectional momentum strategy (no look-ahead).

    12-1 momentum ranked across the universe each month-end → tercile long/short,
    weights lagged one day, minus turnover costs; starts once ≥6 markets are live.
    """
    daily_ret = px.pct_change()
    me = px.resample("ME").last()
    # 12-1 momentum: total return from t-13 to t-1 (skip most recent month).
    mom = me.shift(skip_months) / me.shift(mom_months + skip_months) - 1
    weights_me = mom.apply(_tercile_weights, axis=1)
    w_daily = weights_me.reindex(px.index, method="ffill").shift(1)
    gross_ret = (w_daily * daily_ret).sum(axis=1, min_count=1)
    cost = w_daily.diff().abs().sum(axis=1) * COST_PER_TURNOVER
    net_ret = gross_ret - cost
    breadth = (w_daily != 0).sum(axis=1)
    start = breadth[breadth >= 4].index.min()       # ≥4 active legs (2 long, 2 short)
    return net_ret.loc[start:].fillna(0.0)


def main() -> None:
    tickers = list(UNIVERSE)
    print(f"Fetching {len(tickers)} sector ETFs (max history)…")
    px = fetch_universe(tickers)
    net_ret = xsec_returns(px)

    equity = CAPITAL * (1 + net_ret).cumprod()

    # Long-only top-tercile variant (a tradeable, no-short version)
    daily_ret = px.pct_change()
    me = px.resample("ME").last()
    mom = me.shift(SKIP_MONTHS) / me.shift(MOM_MONTHS + SKIP_MONTHS) - 1

    def _long_only(row: pd.Series) -> pd.Series:
        w = pd.Series(0.0, index=row.index)
        valid = row.dropna()
        if len(valid) < 6:
            return w
        k = len(valid) // 3
        w[valid.sort_values().index[-k:]] = 1.0 / k
        return w

    lo_w = mom.apply(_long_only, axis=1).reindex(px.index, method="ffill").shift(1)
    lo_gross = (lo_w * daily_ret).sum(axis=1, min_count=1)
    lo_cost = lo_w.diff().abs().sum(axis=1) * COST_PER_TURNOVER
    lo_ret = (lo_gross - lo_cost).reindex(net_ret.index).fillna(0.0)
    lo_equity = CAPITAL * (1 + lo_ret).cumprod()

    # Benchmark: SPY buy-and-hold over the same window
    spy = yf.download("SPY", period="max", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    spy = spy.reindex(net_ret.index).ffill().dropna()
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy_eq = CAPITAL * (spy / spy.iloc[0])

    print("\n" + "=" * 60)
    print("  CROSS-SECTIONAL (SECTOR) MOMENTUM — Gate 1 research")
    print(f"  {len(tickers)} sector ETFs, 12-1 momentum, tercile long/short")
    print("=" * 60)
    _report(equity, "X-sec momentum (L/S, dollar-neutral, gross=1)")
    _report(lo_equity, "X-sec momentum (long-only top tercile)")
    _report(spy_eq, "Benchmark: SPY buy & hold")

    # Diversification vs the market
    from backtesting.attribution import factor_attribution
    spy_ret = spy.pct_change().dropna()
    attr = factor_attribution(net_ret, spy_ret)
    print(f"\n  L/S vs SPY:  beta={attr['beta']:.2f}  alpha(ann)={attr['alpha_annual']:+.1%}  "
          f"corr={attr['correlation']:.2f}  R²={attr['r_squared']:.2f}")

    # Diversification vs the VALIDATED trend edge — the real point: is this a
    # second, uncorrelated bet? (portfolio-of-weak-edges)
    try:
        from backtesting.trend_research import fetch_basket, trend_returns
        trend_ret = trend_returns(fetch_basket())
        joined = pd.concat([net_ret, trend_ret], axis=1, join="inner").dropna()
        if len(joined) > 50:
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            print(f"  L/S vs TREND edge:  corr={corr:+.2f}  "
                  f"(overlap {joined.index[0].date()}→{joined.index[-1].date()})")
            print("  → low |corr| ⇒ genuine diversifier; high ⇒ same bet repackaged")
    except Exception as exc:  # research script: keep going if trend fetch fails
        print(f"  (trend-correlation step skipped: {exc})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
