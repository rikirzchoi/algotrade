"""
trend_robustness — Gate-2 stress tests for the diversified trend strategy.

Tries to BREAK the trend result rather than admire it:
  1. Parameter stability — does it work across momentum lookbacks (3–15m), or
     only the 12 we happened to pick? (If only one lookback works, it's overfit.)
  2. Regime breakdown — is the edge spread across the decades, or one lucky era?
  3. Out-of-sample split — does a fixed 12m signal work in BOTH halves of history
     independently?
  4. Ensemble — averaging lookbacks (the robust way to actually trade it).

Run:  .venv/bin/python -m backtesting.trend_robustness
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backtesting.metrics import BacktestMetrics
from backtesting.trend_research import (
    BASKET, CAPITAL, _stats, fetch_basket, trend_returns,
)


def _psr(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 10:
        return float("nan")
    eq = CAPITAL * (1 + ret).cumprod()
    res = SimpleNamespace(
        strategy_id="t", symbol="b",
        start_date=str(ret.index[0].date()), end_date=str(ret.index[-1].date()),
        equity_curve=list(zip(eq.index.to_pydatetime(), eq.to_numpy(float))),
        trades=[],
    )
    return BacktestMetrics(res, CAPITAL, risk_free_rate=0.0).probabilistic_sharpe_ratio


def _row(label: str, ret: pd.Series) -> None:
    ret = ret.dropna()
    if ret.empty:
        return
    ann, sharpe, dd = _stats(ret)
    yrs = len(ret) / 252
    print(f"  {label:22} {yrs:>5.1f}y {ann:>7.1%} {sharpe:>7.2f} {_psr(ret):>6.0%} {dd:>7.1%}")


def _header(title: str) -> None:
    print(f"\n{title}")
    print(f"  {'':22} {'span':>5}  {'ann':>6} {'sharpe':>7} {'PSR':>6} {'maxDD':>7}")


def main() -> None:
    print(f"Fetching {len(BASKET)} markets…")
    px = fetch_basket()
    rets = {m: trend_returns(px, m) for m in (3, 6, 9, 12, 15)}

    _header("1) PARAMETER STABILITY — across momentum lookbacks")
    for m in (3, 6, 9, 12, 15):
        _row(f"{m}-month momentum", rets[m])

    r12 = rets[12]
    _header("2) REGIME BREAKDOWN — fixed 12m, by era")
    for lo, hi in (("2003", "2009"), ("2010", "2015"),
                   ("2016", "2020"), ("2021", "2026")):
        _row(f"{lo}–{hi}", r12.loc[lo:hi])

    _header("3) OUT-OF-SAMPLE SPLIT — fixed 12m, each half alone")
    mid = r12.index[len(r12) // 2]
    _row(f"first half", r12.loc[:mid])
    _row(f"second half", r12.loc[mid:])

    _header("4) ENSEMBLE — average of 3/6/12m (how you'd really trade it)")
    ens = pd.concat([rets[3], rets[6], rets[12]], axis=1).dropna().mean(axis=1)
    _row("3/6/12 ensemble", ens)

    print("\nReading it: robust = positive Sharpe & high PSR across ALL lookbacks,")
    print("ALL eras, and BOTH halves. Overfit = only the 12m / one era works.\n")


if __name__ == "__main__":
    main()
