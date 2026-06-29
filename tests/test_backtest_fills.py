"""
test_backtest_fills — verifies the backtester's conservative fill model:

  * stop is assumed hit first when a bar spans both stop and target
  * market-style exits (stop) take adverse slippage; limit targets do not
  * position sizing uses the same path as live (RiskManager.size_position)

Runnable standalone:

    .venv/bin/python tests/test_backtest_fills.py
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

from backtesting.engine import BacktestEngine
from config import AppConfig
from core.events import BarEvent, Direction, FillEvent, SignalEvent
from risk.manager import RiskManager
from strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")
_T0 = datetime(2025, 1, 2, 16, 0, tzinfo=_ET)


class _OneShotLong(BaseStrategy):
    """Emits a single LONG signal on the first bar with fixed entry/stop/target."""

    def __init__(self, config, entry: float, stop: float, target: float) -> None:
        super().__init__("oneshot", ["TEST"], config)
        self._levels = {"entry": entry, "stop": stop, "target": target}
        self._fired = False

    @property
    def bar_size(self) -> str:
        return "1 day"

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        if self._fired:
            return None
        self._fired = True
        return SignalEvent(
            strategy_id=self.strategy_id, symbol=bar.symbol,
            timestamp=bar.timestamp, direction=Direction.LONG, strength=1.0,
        )

    def on_fill(self, fill: FillEvent) -> None:
        delta = fill.quantity if fill.direction == Direction.LONG else -fill.quantity
        self.update_position(fill.symbol, delta)

    def get_pending_levels(self, symbol: str) -> dict:
        return dict(self._levels)


def _cfg() -> AppConfig:
    return AppConfig()


def _bar(day_offset: int, o: float, h: float, l: float, c: float) -> BarEvent:
    return BarEvent(
        symbol="TEST",
        timestamp=_T0 + timedelta(days=day_offset),
        open=Decimal(str(o)), high=Decimal(str(h)),
        low=Decimal(str(l)), close=Decimal(str(c)),
        volume=1_000_000, bar_size="1 day", vwap=Decimal(str(c)),
    )


def _run(entry, stop, target, exit_bar):
    cfg = _cfg()
    eng = BacktestEngine(cfg, _OneShotLong(cfg, entry, stop, target))
    eng._bars = [_bar(0, entry, entry, entry, entry), exit_bar]
    return eng.run()


def test_stop_assumed_first_on_ambiguous_bar() -> None:
    # Exit bar spans BOTH stop (95) and target (110): low=94, high=115
    res = _run(100.0, 95.0, 110.0, _bar(1, 100, 115, 94, 100))
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "stop", (
        f"ambiguous bar should book the stop, got {res.trades[0].exit_reason}"
    )
    print("PASS test_stop_assumed_first_on_ambiguous_bar")


def test_stop_exit_takes_slippage_and_spread() -> None:
    # Exit bar hits only the stop (low=94), not the target
    res = _run(100.0, 95.0, 110.0, _bar(1, 100, 104, 94, 100))
    t = res.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price < 95.0, f"stop exit should slip below 95, got {t.exit_price}"
    expected = 95.0 * (1 - (0.0005 + 0.0002 / 2))   # slippage + half-spread
    assert abs(t.exit_price - expected) < 1e-6, f"{t.exit_price} != {expected}"
    print("PASS test_stop_exit_takes_slippage_and_spread")


def test_entry_includes_spread() -> None:
    res = _run(100.0, 95.0, 110.0, _bar(1, 100, 111, 96, 105))
    t = res.trades[0]
    expected_entry = 100.0 * (1 + (0.0005 + 0.0002 / 2))  # slippage + half-spread
    assert abs(t.entry_price - expected_entry) < 1e-6, f"{t.entry_price} != {expected_entry}"
    print("PASS test_entry_includes_spread")


def test_short_borrow_cost() -> None:
    from datetime import timedelta as _td
    cfg = _cfg()
    eng = BacktestEngine(cfg, _OneShotLong(cfg, 100.0, 105.0, 90.0))
    trade = __import__("backtesting.engine", fromlist=["BacktestTrade"]).BacktestTrade(
        strategy_id="t", symbol="USO", direction=Direction.SHORT,
        entry_time=_T0, exit_time=None, entry_price=100.0, exit_price=None,
        quantity=10, pnl=None, commission=0.0, exit_reason="",
        stop_price=105.0, target_price=90.0,
    )
    eng._open_trades["USO"] = trade
    eng._close_trade(trade, 98.0, _T0 + _td(days=10), "end_of_data")
    # borrow = entry_price * qty * rate * days/365 = 100*10*0.01*10/365
    expected = 100.0 * 10 * 0.01 * (10 / 365.0)
    assert abs(trade.borrow_cost - expected) < 1e-6, f"{trade.borrow_cost} != {expected}"
    assert trade.borrow_cost > 0
    print(f"PASS test_short_borrow_cost (borrow=${trade.borrow_cost:.4f} for 10 days)")


def test_target_exit_no_slippage() -> None:
    # Exit bar hits only the target (high=111), not the stop
    res = _run(100.0, 95.0, 110.0, _bar(1, 100, 111, 96, 105))
    t = res.trades[0]
    assert t.exit_reason == "target"
    assert abs(t.exit_price - 110.0) < 1e-6, (
        f"limit target should fill at exactly 110, got {t.exit_price}"
    )
    print("PASS test_target_exit_no_slippage")


def test_sizing_matches_live_path() -> None:
    cfg = _cfg()
    res = _run(100.0, 95.0, 110.0, _bar(1, 100, 111, 96, 105))
    qty = res.trades[0].quantity
    # Independently compute what the LIVE risk manager would size
    rm = RiskManager(cfg)
    sig = SignalEvent(strategy_id="oneshot", symbol="TEST", timestamp=_T0,
                      direction=Direction.LONG, strength=1.0)
    live_qty = rm.size_position(sig, 100.0, 95.0)
    assert qty == live_qty, f"backtest qty {qty} != live qty {live_qty}"
    print(f"PASS test_sizing_matches_live_path (qty={qty}, includes macro overlay)")


if __name__ == "__main__":
    tests = [
        test_stop_assumed_first_on_ambiguous_bar,
        test_stop_exit_takes_slippage_and_spread,
        test_target_exit_no_slippage,
        test_sizing_matches_live_path,
        test_entry_includes_spread,
        test_short_borrow_cost,
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
