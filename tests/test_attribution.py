"""
test_attribution — verifies factor attribution (alpha/beta/R²) and portfolio
construction analysis (correlation matrix, inverse-vol weights).

Runnable standalone:

    .venv/bin/python tests/test_attribution.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.attribution import (
    annualised_volatility,
    correlation_matrix,
    factor_attribution,
    inverse_vol_weights,
)

_DATES = pd.date_range("2024-01-01", periods=60, freq="D")


def _series(values) -> pd.Series:
    return pd.Series(values, index=_DATES)


def test_beta_recovered() -> None:
    bench = _series([0.01 * math.sin(i) for i in range(60)])
    strat = bench * 1.5                      # pure 1.5x beta, no alpha
    a = factor_attribution(strat, bench)
    assert abs(a["beta"] - 1.5) < 1e-6, a["beta"]
    assert a["r_squared"] > 0.999, a["r_squared"]
    assert abs(a["alpha_annual"]) < 1e-6, a["alpha_annual"]
    print(f"PASS test_beta_recovered (beta={a['beta']:.2f}, R²={a['r_squared']:.3f})")


def test_alpha_detected() -> None:
    bench = _series([0.01 * math.sin(i) for i in range(60)])
    strat = bench + 0.001                    # beta 1 + 10bps/day alpha
    a = factor_attribution(strat, bench)
    assert abs(a["beta"] - 1.0) < 1e-6, a["beta"]
    assert abs(a["alpha_annual"] - 0.001 * 252) < 1e-6, a["alpha_annual"]
    print(f"PASS test_alpha_detected (alpha_ann={a['alpha_annual']:.1%})")


def test_inverse_vol_weights_favour_low_vol() -> None:
    low = _series([0.01 * math.sin(i) for i in range(60)])    # vol v
    high = _series([0.02 * math.sin(i) for i in range(60)])   # vol 2v
    w = inverse_vol_weights({"low": low, "high": high})
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["low"] - 2 / 3) < 1e-6, w
    assert abs(w["high"] - 1 / 3) < 1e-6, w
    print(f"PASS test_inverse_vol_weights_favour_low_vol ({w})")


def test_correlation_matrix_signs() -> None:
    a = _series([0.01 * math.sin(i) for i in range(60)])
    b = a.copy()             # identical → +1
    c = a * -1.0             # opposite → -1
    cm = correlation_matrix({"a": a, "b": b, "c": c})
    assert abs(cm.loc["a", "b"] - 1.0) < 1e-6
    assert abs(cm.loc["a", "c"] + 1.0) < 1e-6
    print("PASS test_correlation_matrix_signs")


def test_annualised_vol() -> None:
    r = _series([0.01, -0.01] * 30)
    assert annualised_volatility(r) > 0
    assert annualised_volatility(pd.Series(dtype=float)) == 0.0
    print("PASS test_annualised_vol")


def test_insufficient_data_safe() -> None:
    a = factor_attribution(_series([0.0] * 60).iloc[:1], _series([0.0] * 60).iloc[:1])
    assert a["beta"] == 0.0 and a["n_obs"] == 1.0
    print("PASS test_insufficient_data_safe")


if __name__ == "__main__":
    tests = [
        test_beta_recovered,
        test_alpha_detected,
        test_inverse_vol_weights_favour_low_vol,
        test_correlation_matrix_signs,
        test_annualised_vol,
        test_insufficient_data_safe,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            failed += 1
            print(f"ERROR {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
