"""
carry_research — Gate-1 research backtest for the CARRY premium.

Honest scoping note (read first): carry is the return you earn if prices DON'T move
— the yield / roll-down / term-structure component. That component is, by
definition, *removed* from adjusted-price data. So broad cross-asset carry (FX rate
differentials, commodity term structure) is largely **untestable on free
adjusted-price data** — the same "the signal isn't in the data we can get" wall
that parked OFI and broad single-stock momentum. The ONE asset class where carry is
cleanly free-observable is the US rates complex: Yahoo publishes the actual
Treasury yield curve (^IRX 3m, ^TNX 10y, ^TYX 30y) back decades. So this spike
tests *rates carry* specifically, and is honest about being one asset class, not
the full cross-asset carry strategy the literature validates (Koijen-Moskowitz-
Pedersen-Vrugt 2018).

Thesis: a positive term spread (long yield > funding rate) means a financed long
bond earns positive carry + roll-down; an inverted curve means negative carry and
historically precedes duration pain. Harvest the premium when carry is positive,
step aside (to cash-like SHY) when it is not. Counterparty: investors who hold
duration regardless of carry (liability-matchers, index funds).

PRE-COMMITTED DESIGN (fixed before results, per TRADING_RULES.md §10):
  * Variant A — duration timing: long TLT when the 10y-3m term spread > 0 at
    month-end, else hold SHY (cash-like). Always fully invested.
  * Variant B — cross-sectional rates carry: each month, net carry of each bucket
    = its yield - 3m rate (SHY≈0, IEF=10y-3m, TLT=30y-3m); inverse-vol weight the
    POSITIVE-carry buckets, gross=1 long-only; all-inverted → SHY.
  * rebalance month-end; weights lagged 1 day; 10 bps per unit turnover.
NO parameter sweeps. Signs/units cancel in the spread, so yield-quote scaling is
irrelevant to the signal. Reports correlation to the validated trend edge + SPY.

Run:  .venv/bin/python -m backtesting.carry_research
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting.metrics import BacktestMetrics

CAPITAL = 100_000.0
COST_PER_TURNOVER = 0.0010

RATES = {"^IRX": "3m", "^TNX": "10y", "^TYX": "30y"}
ETFS = ["SHY", "IEF", "TLT"]                  # 1-3y, 7-10y, 20y+ duration buckets


def _metrics(equity: pd.Series, label: str) -> BacktestMetrics:
    res = SimpleNamespace(
        strategy_id=label, symbol="rates",
        start_date=str(equity.index[0].date()), end_date=str(equity.index[-1].date()),
        equity_curve=list(zip(equity.index.to_pydatetime(), equity.to_numpy(float))),
        trades=[],
    )
    return BacktestMetrics(res, CAPITAL, risk_free_rate=0.0)


def _stats(r: pd.Series) -> tuple[float, float, float]:
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


def _close(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, period="max", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    return raw.sort_index()


def _net_from_weights(w_me: pd.DataFrame, px: pd.DataFrame) -> pd.Series:
    """Monthly weight matrix → daily net return (lag 1d, turnover cost)."""
    daily_ret = px.pct_change()
    w_daily = w_me.reindex(px.index, method="ffill").shift(1)
    gross = (w_daily * daily_ret).sum(axis=1, min_count=1)
    cost = w_daily.diff().abs().sum(axis=1) * COST_PER_TURNOVER
    net = gross - cost
    breadth = (w_daily.abs() > 0).sum(axis=1)
    start = breadth[breadth >= 1].index.min()
    return net.loc[start:].fillna(0.0)


def carry_returns(px: pd.DataFrame, yld: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Returns of Variant A (duration timing) and Variant B (x-sec rates carry)."""
    px_me = px.resample("ME").last()
    yld_me = yld.resample("ME").last()

    # Net carry per bucket = bucket yield - 3m funding rate.
    carry = pd.DataFrame(index=yld_me.index)
    carry["SHY"] = 0.0
    carry["IEF"] = yld_me["^TNX"] - yld_me["^IRX"]
    carry["TLT"] = yld_me["^TYX"] - yld_me["^IRX"]
    carry = carry.reindex(columns=ETFS)

    # ---- Variant A: long TLT when 10y-3m term spread > 0, else SHY ----
    spread = (yld_me["^TNX"] - yld_me["^IRX"])
    wA = pd.DataFrame(0.0, index=px_me.index, columns=ETFS)
    long_tlt = spread.reindex(px_me.index) > 0
    wA.loc[long_tlt, "TLT"] = 1.0
    wA.loc[~long_tlt, "SHY"] = 1.0

    # ---- Variant B: inverse-vol weight the positive-carry buckets (long-only) ----
    daily_ret = px.pct_change()
    ann_vol_me = (daily_ret.ewm(span=63, min_periods=20).std() * np.sqrt(252)
                  ).resample("ME").last()
    pos_carry = carry.reindex(px_me.index).clip(lower=0.0)
    raw_w = pos_carry / ann_vol_me.reindex(px_me.index).replace(0.0, np.nan)
    tot = raw_w.sum(axis=1)
    wB = raw_w.div(tot, axis=0)
    flat = tot.fillna(0.0) <= 0                    # all carry ≤ 0 → cash (SHY)
    wB.loc[flat, :] = 0.0
    wB.loc[flat, "SHY"] = 1.0
    wB = wB.fillna(0.0)

    return _net_from_weights(wA, px), _net_from_weights(wB, px)


def main() -> None:
    print(f"Fetching rates curve {list(RATES)} and ETFs {ETFS}…")
    yld = _close(list(RATES)).dropna(how="all").ffill()
    px = _close(ETFS).dropna(how="all")
    retA, retB = carry_returns(px, yld)

    eqA = CAPITAL * (1 + retA).cumprod()
    eqB = CAPITAL * (1 + retB).cumprod()

    # Benchmarks over the common window
    idx = retB.index
    tlt = px["TLT"].reindex(idx).ffill().dropna()
    tlt_eq = CAPITAL * (tlt / tlt.iloc[0])
    ief = px["IEF"].reindex(idx).ffill().dropna()
    ief_eq = CAPITAL * (ief / ief.iloc[0])

    print("\n" + "=" * 60)
    print("  RATES CARRY — Gate 1 research (one asset class; see scoping note)")
    print("=" * 60)
    _report(eqA, "A. Duration timing (long TLT if 10y-3m>0, else SHY)")
    _report(eqB, "B. X-sec rates carry (inv-vol, positive-carry buckets)")
    _report(tlt_eq, "Benchmark: TLT buy & hold")
    _report(ief_eq, "Benchmark: IEF buy & hold")

    from backtesting.attribution import factor_attribution
    spy = _close(["SPY"])["SPY"].reindex(idx).ffill()
    spy_ret = spy.pct_change().dropna()
    for ret, lbl in ((retA, "A"), (retB, "B")):
        attr = factor_attribution(ret, spy_ret)
        print(f"\n  {lbl} vs SPY:  beta={attr['beta']:.2f}  "
              f"alpha(ann)={attr['alpha_annual']:+.1%}  corr={attr['correlation']:.2f}")

    # Diversification vs the validated trend edge
    try:
        from backtesting.trend_research import fetch_basket, trend_returns
        trend_ret = trend_returns(fetch_basket())
        for ret, lbl in ((retA, "A"), (retB, "B")):
            j = pd.concat([ret, trend_ret], axis=1, join="inner").dropna()
            if len(j) > 50:
                print(f"  {lbl} vs TREND edge:  corr={j.iloc[:,0].corr(j.iloc[:,1]):+.2f}")
    except Exception as exc:
        print(f"  (trend-correlation step skipped: {exc})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
