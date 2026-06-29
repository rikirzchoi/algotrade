"""
test_portfolio_backtest — verifies the multi-strategy portfolio backtest
replays the real live stack: multiple strategies trade together, the
duplicate-position and correlation filters engage, the portfolio-level kill
switch fires and blocks further entries, and bar-size routing is isolated.

Runnable standalone:

    .venv/bin/python tests/test_portfolio_backtest.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.portfolio import PortfolioBacktestEngine
from config import AppConfig
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")
_T0 = datetime(2024, 1, 2, 10, 0, tzinfo=_ET)


def _cfg(**risk_overrides) -> AppConfig:
    base = AppConfig()
    if risk_overrides:
        return replace(base, risk=replace(base.risk, **risk_overrides))
    return base


def _bar(symbol, minute_offset, o, h, l, c, bar_size="15 mins") -> BarEvent:
    return BarEvent(
        symbol=symbol, timestamp=_T0 + timedelta(minutes=minute_offset),
        open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(l)),
        close=Decimal(str(c)), volume=1_000_000, bar_size=bar_size,
        vwap=Decimal(str(c)),
    )


class Scripted(BaseStrategy):
    """Fires one LONG on the first matching bar with fixed levels."""

    def __init__(self, config, sid, symbol, entry, stop, target, bar_size="15 mins"):
        super().__init__(sid, [symbol], config)
        self._bs = bar_size
        self._levels = {"entry": entry, "stop": stop, "target": target}
        self._fired = False
        self.on_bar_calls = 0

    @property
    def bar_size(self) -> str:
        return self._bs

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        self.on_bar_calls += 1
        if self._fired:
            return None
        self._fired = True
        return SignalEvent(strategy_id=self.strategy_id, symbol=bar.symbol,
                           timestamp=bar.timestamp, direction=Direction.LONG, strength=1.0)

    def on_fill(self, fill: FillEvent) -> None:
        delta = fill.quantity if fill.direction == Direction.LONG else -fill.quantity
        self.update_position(fill.symbol, delta)

    def get_pending_levels(self, symbol: str) -> dict:
        return dict(self._levels)


def test_two_strategies_two_symbols_both_trade() -> None:
    cfg = _cfg()
    eng = PortfolioBacktestEngine(cfg, [
        Scripted(cfg, "stratA", "MSFT", 100, 95, 110),
        Scripted(cfg, "stratB", "GLD", 50, 48, 55),
    ])
    eng.add_series([_bar("MSFT", 0, 100, 100, 100, 100),
                    _bar("MSFT", 60, 100, 111, 99, 105)])   # hits target 110
    eng.add_series([_bar("GLD", 1, 50, 50, 50, 50),
                    _bar("GLD", 61, 50, 56, 49, 55)])        # hits target 55
    res = eng.run()
    assert len(res.trades) == 2, f"expected 2 trades, got {len(res.trades)}"
    assert set(res.per_strategy) == {"stratA", "stratB"}
    assert all(t.exit_reason == "target" for t in res.trades)
    print("PASS test_two_strategies_two_symbols_both_trade")


def test_duplicate_position_filter_blocks_second_strategy() -> None:
    cfg = _cfg()
    # Two strategies competing for the SAME symbol on the same bar
    eng = PortfolioBacktestEngine(cfg, [
        Scripted(cfg, "first", "AMZN", 100, 95, 110),
        Scripted(cfg, "second", "AMZN", 100, 95, 110),
    ])
    eng.add_series([_bar("AMZN", 0, 100, 100, 100, 100),
                    _bar("AMZN", 60, 100, 111, 99, 105)])
    res = eng.run()
    assert len(res.trades) == 1, f"only one AMZN position allowed, got {len(res.trades)}"
    assert res.signals_rejected >= 1
    assert any("already in position" in k for k in res.rejection_reasons)
    print("PASS test_duplicate_position_filter_blocks_second_strategy")


def test_correlation_filter_blocks_third_risk_on() -> None:
    cfg = _cfg()   # risk_on=(QQQ,MSFT,NVDA,AMZN), max_risk_on_positions=2
    eng = PortfolioBacktestEngine(cfg, [
        Scripted(cfg, "s_msft", "MSFT", 100, 95, 110),
        Scripted(cfg, "s_nvda", "NVDA", 100, 95, 110),
        Scripted(cfg, "s_amzn", "AMZN", 100, 95, 110),
    ])
    # Trigger bars, no exits — all stay open so the 3rd hits the concentration cap
    eng.add_series([_bar("MSFT", 0, 100, 100, 100, 100)])
    eng.add_series([_bar("NVDA", 1, 100, 100, 100, 100)])
    eng.add_series([_bar("AMZN", 2, 100, 100, 100, 100)])
    res = eng.run()
    opened = {t.symbol for t in res.trades}
    assert "AMZN" not in opened, "3rd correlated risk-on long should be blocked"
    assert any("correlation" in k for k in res.rejection_reasons), res.rejection_reasons
    print("PASS test_correlation_filter_blocks_third_risk_on")


def test_portfolio_kill_switch_blocks_later_entries() -> None:
    cfg = _cfg(max_daily_loss_usd=100.0)
    eng = PortfolioBacktestEngine(cfg, [
        Scripted(cfg, "loser", "MSFT", 100, 90, 130),   # wide stop → big $ loss
        Scripted(cfg, "later", "GLD", 50, 48, 55),
    ])
    # MSFT triggers, then stops out hard → daily loss breached → halt
    eng.add_series([_bar("MSFT", 0, 100, 100, 100, 100),
                    _bar("MSFT", 60, 100, 100, 85, 88)])   # low 85 < stop 90
    # GLD signal comes AFTER the halt
    eng.add_series([_bar("GLD", 120, 50, 50, 50, 50)])
    res = eng.run()
    assert res.halted, "portfolio kill switch should have fired"
    assert "GLD" not in {t.symbol for t in res.trades}, "no entries allowed after halt"
    assert any("kill switch" in k.lower() for k in res.rejection_reasons), res.rejection_reasons
    print(f"PASS test_portfolio_kill_switch_blocks_later_entries ({res.halt_info})")


def test_bar_size_routing_isolated() -> None:
    cfg = _cfg()
    strat = Scripted(cfg, "s15", "AMZN", 100, 95, 110, bar_size="15 mins")
    eng = PortfolioBacktestEngine(cfg, [strat])
    # Feed a 5-min AMZN bar (wrong size) and a 15-min AMZN bar (right size)
    eng.add_series([_bar("AMZN", 0, 100, 100, 100, 100, bar_size="5 mins")])
    eng.add_series([_bar("AMZN", 1, 100, 100, 100, 100, bar_size="15 mins")])
    eng.run()
    assert strat.on_bar_calls == 1, (
        f"strategy should only receive its own bar size, got {strat.on_bar_calls} calls"
    )
    print("PASS test_bar_size_routing_isolated")


if __name__ == "__main__":
    tests = [
        test_two_strategies_two_symbols_both_trade,
        test_duplicate_position_filter_blocks_second_strategy,
        test_correlation_filter_blocks_third_risk_on,
        test_portfolio_kill_switch_blocks_later_entries,
        test_bar_size_routing_isolated,
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
