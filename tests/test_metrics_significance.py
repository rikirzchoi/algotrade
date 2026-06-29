"""
test_metrics_significance — verifies the statistical-honesty additions to
BacktestMetrics: no absurd annualisation of short windows, and significance
measures (t-stat, Sharpe CI, PSR, is_significant) behave sensibly.

Runnable standalone:

    .venv/bin/python tests/test_metrics_significance.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.metrics import BacktestMetrics

_ET = ZoneInfo("America/New_York")
_T0 = datetime(2024, 1, 2, 16, 0, tzinfo=_ET)
_CAPITAL = 100_000.0


def _result(daily_returns: list[float], n_trades: int):
    """Build a minimal BacktestResult-like object from a daily return series."""
    eq = _CAPITAL
    curve = [(_T0, eq)]
    for i, r in enumerate(daily_returns, start=1):
        eq *= (1 + r)
        curve.append((_T0 + timedelta(days=i), eq))
    trades = [SimpleNamespace(pnl=100.0) for _ in range(n_trades)]
    return SimpleNamespace(
        strategy_id="t", symbol="TEST",
        start_date="2024-01-02", end_date="2024-12-31",
        equity_curve=curve, trades=trades,
    )


def test_short_window_not_annualised() -> None:
    # 10 days of +0.5%/day — CAGR would extrapolate absurdly
    m = BacktestMetrics(_result([0.005] * 10, n_trades=3), _CAPITAL)
    assert abs(m.annualised_return - m.total_return) < 1e-9, (
        f"short window should report total return, not extrapolated CAGR "
        f"(ann={m.annualised_return}, total={m.total_return})"
    )
    assert m.annualised_return < 0.10, "10 days @0.5% must not read as a huge annual %"
    print("PASS test_short_window_not_annualised")


def test_small_sample_not_significant() -> None:
    m = BacktestMetrics(_result([0.01, -0.005, 0.008, 0.002], n_trades=2), _CAPITAL)
    assert not m.is_significant, "4 obs / 2 trades must never read as significant"
    assert 0.0 <= m.probabilistic_sharpe_ratio <= 1.0
    print("PASS test_small_sample_not_significant")


def test_strong_long_sample_is_significant() -> None:
    # 300 days, steady positive drift with real variance → clearly significant
    returns = [0.001 + 0.005 * math.sin(i) for i in range(300)]
    m = BacktestMetrics(_result(returns, n_trades=60), _CAPITAL)
    assert m.num_return_observations >= 250
    assert m.sharpe_tstat > 2.0, f"t-stat should clear 2, got {m.sharpe_tstat:.2f}"
    assert m.probabilistic_sharpe_ratio > 0.95, m.probabilistic_sharpe_ratio
    assert m.is_significant, "300 obs / 60 trades with positive Sharpe should be significant"
    lo, hi = m.sharpe_ci_95
    assert lo <= hi
    print(f"PASS test_strong_long_sample_is_significant "
          f"(t={m.sharpe_tstat:.1f}, PSR={m.probabilistic_sharpe_ratio:.2f}, "
          f"CI=[{lo:.2f},{hi:.2f}])")


def test_ci_straddles_zero_for_weak_edge() -> None:
    # Near-zero mean return → CI should straddle zero (no demonstrated edge)
    returns = [0.005 * math.sin(i) for i in range(120)]   # mean ~0
    m = BacktestMetrics(_result(returns, n_trades=40), _CAPITAL)
    lo, hi = m.sharpe_ci_95
    assert lo < 0 < hi, f"weak edge CI should straddle zero, got [{lo:.2f},{hi:.2f}]"
    assert not m.is_significant
    print(f"PASS test_ci_straddles_zero_for_weak_edge (CI=[{lo:.2f},{hi:.2f}])")


if __name__ == "__main__":
    tests = [
        test_short_window_not_annualised,
        test_small_sample_not_significant,
        test_strong_long_sample_is_significant,
        test_ci_straddles_zero_for_weak_edge,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
