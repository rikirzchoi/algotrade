"""
trend_research — Gate-1 research backtest for DIVERSIFIED time-series momentum
(trend following) across a basket of liquid ETF proxies spanning asset classes.

Thesis: assets trending over the past ~12 months continue over the next ~1 month
(behavioural underreaction + crisis-alpha premium). Edge comes from averaging
many *uncorrelated* trend bets — which is why a 2-symbol GLD/USO version failed.

Method (vectorised, conservative, no look-ahead):
  * signal = sign of trailing 12-month total return, set at each month-end
  * inverse-volatility weights (risk parity), gross normalised to 1 (unlevered)
  * weights lagged 1 day before applying to daily returns (no same-day look-ahead)
  * turnover-based transaction costs
  * daily P&L → daily equity curve → BacktestMetrics (Sharpe/PSR/drawdown)
  * also a vol-targeted (~12%/yr) variant using trailing vol

Run:  .venv/bin/python -m backtesting.trend_research
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting.metrics import BacktestMetrics

CAPITAL = 100_000.0
COST_PER_TURNOVER = 0.0010          # 10 bps per unit of weight traded
MOM_MONTHS = 12                     # trailing-return lookback
TARGET_VOL = 0.12                   # for the vol-targeted variant

BASKET = {
    "SPY": "US large-cap", "QQQ": "US tech", "IWM": "US small-cap",
    "EFA": "Dev. ex-US", "EEM": "Emerging mkts",
    "TLT": "Long Treasuries", "IEF": "7-10y Treasuries", "LQD": "IG credit",
    "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities", "USO": "Oil",
    "VNQ": "US REITs", "UUP": "US dollar",
}


def _metrics(equity: pd.Series, label: str) -> BacktestMetrics:
    res = SimpleNamespace(
        strategy_id=label, symbol="basket",
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


def fetch_basket() -> pd.DataFrame:
    """Download adjusted daily closes for the whole basket (max history)."""
    raw = yf.download(list(BASKET), period="max", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    return raw.dropna(how="all").sort_index()


def trend_returns(px: pd.DataFrame, mom_months: int = MOM_MONTHS) -> pd.Series:
    """Daily net returns of the diversified trend strategy (no look-ahead).

    sign(trailing mom_months return) × inverse-vol weight, gross normalised to 1,
    weights lagged one day, minus turnover costs; starts once ≥5 markets are live.
    """
    daily_ret = px.pct_change()
    me = px.resample("ME").last()
    signal = np.sign(me.pct_change(mom_months))
    ann_vol_me = (daily_ret.ewm(span=63, min_periods=20).std() * np.sqrt(252)
                  ).resample("ME").last()
    raw_w = signal / ann_vol_me.replace(0.0, np.nan)
    weights_me = raw_w.div(raw_w.abs().sum(axis=1), axis=0)
    w_daily = weights_me.reindex(px.index, method="ffill").shift(1)
    gross_ret = (w_daily * daily_ret).sum(axis=1, min_count=1)
    cost = w_daily.diff().abs().sum(axis=1) * COST_PER_TURNOVER
    net_ret = gross_ret - cost
    breadth = w_daily.notna().sum(axis=1)
    start = breadth[breadth >= 5].index.min()
    return net_ret.loc[start:].fillna(0.0)


def main() -> None:
    print(f"Fetching {len(BASKET)} markets (max history)…")
    px = fetch_basket()
    net_ret = trend_returns(px)

    equity = CAPITAL * (1 + net_ret).cumprod()

    # Vol-targeted variant (trailing vol, lagged → no look-ahead)
    trail_vol = net_ret.ewm(span=63, min_periods=20).std() * np.sqrt(252)
    lev = (TARGET_VOL / trail_vol).shift(1).clip(upper=3.0).fillna(0.0)
    vt_equity = CAPITAL * (1 + net_ret * lev).cumprod()

    # Benchmark: SPY buy-and-hold over the same window
    spy = px["SPY"].reindex(net_ret.index).dropna()
    spy_eq = CAPITAL * (spy / spy.iloc[0])

    print("\n" + "=" * 60)
    print("  DIVERSIFIED TREND FOLLOWING — Gate 1 research")
    print(f"  {len(tickers)} markets, 12m momentum, inverse-vol, long/short")
    print("=" * 60)
    _report(equity, "Trend (unlevered, gross=1)")
    _report(vt_equity, f"Trend (vol-targeted ~{TARGET_VOL:.0%})")
    _report(spy_eq, "Benchmark: SPY buy & hold")

    # Correlation of the strategy to the market (diversification value)
    from backtesting.attribution import factor_attribution
    spy_ret = spy.pct_change().dropna()
    attr = factor_attribution(net_ret, spy_ret)
    print(f"\n  vs SPY:  beta={attr['beta']:.2f}  alpha(ann)={attr['alpha_annual']:+.1%}  "
          f"corr={attr['correlation']:.2f}  R²={attr['r_squared']:.2f}")

    # ---- Blend: capital-weighted SPY + unlevered trend ----
    spy_d = px["SPY"].pct_change().reindex(net_ret.index).fillna(0.0)
    print("\n" + "-" * 60)
    print("  BLEND: w·SPY + (1-w)·Trend  (unlevered, same window)")
    print(f"  {'w(SPY)':>7} {'ann':>8} {'sharpe':>8} {'maxDD':>8}")
    best = (None, -9.9)
    for i in range(0, 11):
        w = i / 10
        ann, sharpe, dd = _stats(w * spy_d + (1 - w) * net_ret)
        print(f"  {w:>7.0%} {ann:>8.1%} {sharpe:>8.2f} {dd:>8.1%}")
        if sharpe > best[1]:
            best = (w, sharpe)
    print(f"\n  → best risk-adjusted blend: {best[0]:.0%} SPY / "
          f"{1-best[0]:.0%} trend (Sharpe {best[1]:.2f})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
