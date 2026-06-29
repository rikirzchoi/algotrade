"""
vrp_research — Gate-1 research backtest for the VOLATILITY RISK PREMIUM (VRP).

Thesis: option-implied volatility systematically exceeds subsequently-realised
volatility — sellers of volatility/variance earn a premium for bearing the risk of
volatility spikes. It is a REAL, well-documented premium (unlike the momentum/carry
spikes that showed no edge). The catch is its SHAPE: steady small premiums punctuated
by catastrophic losses — negative skew, "picking up pennies in front of a steamroller."
So the honest test is not "is the Sharpe high?" but "what is the LEFT TAIL, what do
realistic costs do to it, and can we even access it?"  Counterparty: hedgers who
overpay for downside insurance.

Construction (free data only — ^VIX implied, ^GSPC realised): a rolled SHORT
1-month variance swap. Each month-end set the strike to the prevailing implied
variance K = (VIX/100)²; over the next month the short-variance position pays the
realised variance RV² of daily SPX log returns. Monthly P&L (variance points):
        pnl_m = K_{prev}  −  RV_m²            (short variance: win when implied>realised)
Strike is set on PRIOR month-end data → no look-ahead. Scaled to ~15% annual vol
(Sharpe-invariant; affects only drawdown readability). Reported GROSS and NET of a
realistic variance-swap/VIX-future round-trip cost (COST_VOLPTS per month), because
the premium is only ~1.4 vol points and costs eat a large slice of it.

PRE-COMMITTED: one configuration, no sweep. The report foregrounds the TAIL — skew,
worst month, max drawdown — and the skew/kurtosis-adjusted PSR, because raw Sharpe
flatters a negatively-skewed strategy.

IMPLEMENTABILITY NOTE: the clean instrument (variance swap) is institutional OTC, not
retail-accessible. Tradeable retail proxies are lethal — XIV (short-vol ETN)
TERMINATED −96% in a day on 2018-02-05 ("Volmageddon"); short VXX/SVXY carry the same
path-dependency. This backtest measures the PREMIUM, not a tradeable product.

Run:  .venv/bin/python -m backtesting.vrp_research
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting.metrics import BacktestMetrics

CAPITAL = 100_000.0
TARGET_VOL = 0.15
COST_VOLPTS = 0.5        # realistic round-trip slippage on the strike, vol points/month


def _close(ticker: str) -> pd.Series:
    s = yf.download(ticker, period="max", interval="1d",
                    auto_adjust=True, progress=False)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def vrp_monthly(cost_volpts: float = 0.0) -> pd.Series:
    """Monthly returns of the rolled short-variance-swap VRP proxy (vol-targeted).

    cost_volpts: round-trip transaction cost per month, in vol points (0 = gross).
    """
    spx = _close("^GSPC")
    vix = _close("^VIX")
    df = pd.concat([spx.rename("spx"), vix.rename("vix")], axis=1).dropna()
    log_ret = np.log(df["spx"]).diff()

    strike = ((df["vix"].resample("ME").last() / 100.0) ** 2).shift(1)  # implied var, lagged
    realised = (log_ret.resample("ME").std() * np.sqrt(252)) ** 2       # realised var/mo
    pnl = (strike - realised).dropna()                                  # variance points

    if cost_volpts:
        k_vol = (df["vix"].resample("ME").last() / 100.0).reindex(pnl.index)
        pnl = pnl - 2 * k_vol * (cost_volpts / 100.0)                   # d(V²)=2V·dV

    scale = TARGET_VOL / (pnl.std() * np.sqrt(12))                      # Sharpe-invariant
    return pnl * scale


def _report(ret: pd.Series, label: str, ppy: int = 12) -> None:
    equity = CAPITAL * (1 + ret).cumprod()
    res = SimpleNamespace(
        strategy_id=label, symbol="vrp",
        start_date=str(equity.index[0].date()), end_date=str(equity.index[-1].date()),
        equity_curve=list(zip(equity.index.to_pydatetime(), equity.to_numpy(float))),
        trades=[],
    )
    m = BacktestMetrics(res, CAPITAL, risk_free_rate=0.0)
    sharpe = ret.mean() / ret.std() * np.sqrt(ppy)
    ann = (equity.iloc[-1] / equity.iloc[0]) ** (ppy / max(len(equity), 1)) - 1
    print(f"\n  {label}")
    print(f"    period          {equity.index[0].date()} → {equity.index[-1].date()}")
    print(f"    ann. return     {ann:+.1%}")
    print(f"    Sharpe          {sharpe:.2f}")
    print(f"    PSR P[SR>0]     {m.probabilistic_sharpe_ratio:.0%}  (skew/kurt-adjusted)")
    print(f"    max drawdown    {m.max_drawdown:.1%}")
    print(f"    --- the tail (the whole point) ---")
    print(f"    monthly skew    {ret.skew():+.2f}   (a healthy edge is ≥ 0)")
    print(f"    worst month     {ret.min():+.1%}")


def main() -> None:
    print("Fetching ^GSPC and ^VIX (max history)…")
    gross = vrp_monthly(cost_volpts=0.0)
    net = vrp_monthly(cost_volpts=COST_VOLPTS)

    print("\n" + "=" * 60)
    print("  VOLATILITY RISK PREMIUM — Gate 1 research")
    print("  (rolled short 1-month variance swap; free-data proxy)")
    print("=" * 60)
    _report(gross, "GROSS short-variance (no costs)")
    _report(net, f"NET of {COST_VOLPTS} vol-pt/mo round-trip cost")

    print("\n  5 worst months (gross):")
    for d, v in gross.nsmallest(5).items():
        print(f"    {d.date()}  {v:+.1%}")

    # average premium in vol points (why costs bite)
    vix = _close("^VIX")
    spx = _close("^GSPC")
    rv = (np.log(spx).diff().std() * np.sqrt(252)) * 100
    print(f"\n  avg implied vol {vix.mean():.1f}  vs realised {rv:.1f}  "
          f"→ VRP ≈ {vix.mean()-rv:.1f} vol pts  (cost {COST_VOLPTS}/mo eats a big slice)")

    from backtesting.attribution import factor_attribution
    spy = _close("SPY").pct_change().resample("ME").sum().dropna()
    attr = factor_attribution(net.reindex(spy.index).dropna(),
                              spy.reindex(net.index).dropna())
    print(f"\n  vs SPY:  beta={attr['beta']:.2f}  corr={attr['correlation']:.2f}  "
          f"(short-vol is LONG equity-crash risk → wrong sign for a diversifier)")
    try:
        from backtesting.trend_research import fetch_basket, trend_returns
        tr = (1 + trend_returns(fetch_basket())).resample("ME").prod() - 1
        j = pd.concat([net, tr], axis=1, join="inner").dropna()
        if len(j) > 24:
            print(f"  vs TREND edge:  corr={j.iloc[:,0].corr(j.iloc[:,1]):+.2f}  "
                  f"(trend is LONG vol/crisis-alpha — VRP is the opposite bet)")
    except Exception as exc:
        print(f"  (trend-correlation step skipped: {exc})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
