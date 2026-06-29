"""
run_backtest — run each strategy (and the full portfolio) over downloaded data
and print Gate-1 verdicts (significance via PSR), plus benchmark attribution.

Prereq:  .venv/bin/python -m backtesting.fetch_data
Run:     .venv/bin/python -m backtesting.run_backtest
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtesting.attribution import daily_returns_from_equity_curve
from backtesting.data_loader import parse_ohlcv_csv
from backtesting.portfolio import PortfolioBacktestEngine
from config import AppConfig
from strategies.bollinger_reversion import BollingerReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy

DATA = Path(__file__).resolve().parent.parent / "data"
CFG = AppConfig.from_env()

# strategy factory → list of (symbol, csv filename, bar_size)
SPECS = {
    "momentum_breakout": (MomentumBreakoutStrategy,
        [("QQQ", "QQQ_1d.csv", "1 day")]),
    "trend_following": (TrendFollowingStrategy,
        [("GLD", "GLD_4h.csv", "4 hours"), ("USO", "USO_4h.csv", "4 hours")]),
    "bollinger_reversion": (BollingerReversionStrategy,
        [("MSFT", "MSFT_15m.csv", "15 mins"), ("NVDA", "NVDA_15m.csv", "15 mins"),
         ("AMZN", "AMZN_15m.csv", "15 mins")]),
    "opening_range_breakout": (OpeningRangeBreakoutStrategy,
        [("AMZN", "AMZN_5m.csv", "5 mins")]),
}


def _benchmark() -> pd.Series:
    bars = parse_ohlcv_csv(DATA / "SPY_1d.csv", "SPY", "1 day")
    s = pd.Series({b.timestamp: float(b.close) for b in bars})
    s.index = pd.to_datetime(s.index)
    return s.pct_change().dropna()


def _load(eng: PortfolioBacktestEngine, specs) -> bool:
    ok = False
    for sym, fname, bar_size in specs:
        path = DATA / fname
        if not path.exists():
            print(f"   (missing {fname} — skipped)")
            continue
        eng.load_csv(path, sym, bar_size)
        ok = True
    return ok


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    bench = _benchmark()

    # --- each strategy on its own ---
    all_strategies, all_specs = [], []
    for name, (factory, specs) in SPECS.items():
        eng = PortfolioBacktestEngine(CFG, [factory(CFG)])
        if not _load(eng, specs):
            continue
        eng.run().summary(benchmark_returns=bench)
        all_strategies.append(factory(CFG))
        all_specs += specs

    # --- the full portfolio together ---
    print("\n########## FULL PORTFOLIO (all strategies, shared account) ##########")
    port = PortfolioBacktestEngine(CFG, all_strategies)
    _load(port, all_specs)
    port.run().summary(benchmark_returns=bench)


if __name__ == "__main__":
    main()
