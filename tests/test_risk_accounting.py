"""
test_risk_accounting — verifies that realised PnL is booked on round-trip
closes and that the daily-loss / drawdown kill switches actually fire.

Runnable standalone (no pytest required):

    .venv/bin/python tests/test_risk_accounting.py

Covers the bug where RiskManager.record_close() was never called, leaving the
account-level kill switches inert in live trading.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from core.engine import TradingEngine
from core.events import Direction, FillEvent, SignalEvent
from risk.manager import RiskManager

_ET = ZoneInfo("America/New_York")


def _cfg(max_daily_loss: float = 500.0, capital: float = 100_000.0) -> AppConfig:
    base = AppConfig()
    tmp_db = Path(tempfile.mkdtemp()) / "test.db"
    return replace(
        base,
        risk=replace(base.risk, max_daily_loss_usd=max_daily_loss, capital=capital),
        database=replace(base.database, db_path=tmp_db),
    )


def _fill(symbol: str, direction: Direction, qty: int, price: float) -> FillEvent:
    """direction here is the EXECUTION side: LONG = bought, SHORT = sold."""
    return FillEvent(
        strategy_id="test",
        symbol=symbol,
        timestamp=datetime.now(tz=_ET),
        direction=direction,
        quantity=qty,
        fill_price=Decimal(str(price)),
        commission=Decimal("0.0"),
    )


def _signal(symbol: str, direction: Direction) -> SignalEvent:
    return SignalEvent(
        strategy_id="test",
        symbol=symbol,
        timestamp=datetime.now(tz=_ET),
        direction=direction,
        strength=1.0,
    )


# ---------------------------------------------------------------------------
# RiskManager unit tests
# ---------------------------------------------------------------------------

def test_record_close_updates_pnl_and_halts() -> None:
    rm = RiskManager(_cfg(max_daily_loss=500.0))
    assert rm.daily_pnl == 0.0
    assert not rm.is_halted

    # Losing trade that breaches the daily loss limit
    pnl = rm.record_close("NVDA", entry_price=100.0, exit_price=94.0,
                          quantity=100, direction=Direction.LONG)
    assert pnl == -600.0, f"expected -600, got {pnl}"
    assert rm.daily_pnl == -600.0
    assert rm.is_halted, "kill switch should trip after exceeding daily loss"

    # Once halted, new entry signals are rejected
    ok, reason = rm.approve_signal(_signal("AMZN", Direction.LONG), {})
    assert not ok and "halt" in reason.lower(), reason
    print("PASS test_record_close_updates_pnl_and_halts")


def test_short_pnl_sign() -> None:
    rm = RiskManager(_cfg())
    pnl = rm.record_close("USO", entry_price=100.0, exit_price=95.0,
                          quantity=10, direction=Direction.SHORT)
    assert pnl == 50.0, f"short win should be +50, got {pnl}"
    assert rm.daily_pnl == 50.0
    print("PASS test_short_pnl_sign")


# ---------------------------------------------------------------------------
# Engine ledger / round-trip pairing tests
# ---------------------------------------------------------------------------

def test_engine_long_round_trip() -> None:
    eng = TradingEngine(_cfg())
    # Entry: BOT 10 @ 100  → opens a LONG
    eng._account_for_fill(_fill("MSFT", Direction.LONG, 10, 100.0))
    assert "MSFT" in eng._open_positions
    assert eng._risk.daily_pnl == 0.0, "no PnL until the position closes"

    # Exit: SLD 10 @ 105  → closes the LONG for +50
    eng._account_for_fill(_fill("MSFT", Direction.SHORT, 10, 105.0))
    assert "MSFT" not in eng._open_positions, "ledger should be flat after close"
    assert eng._risk.daily_pnl == 50.0, f"expected +50, got {eng._risk.daily_pnl}"
    print("PASS test_engine_long_round_trip")


def test_engine_short_round_trip() -> None:
    eng = TradingEngine(_cfg())
    # Entry: SLD 10 @ 100 → opens a SHORT
    eng._account_for_fill(_fill("GLD", Direction.SHORT, 10, 100.0))
    assert eng._open_positions["GLD"].side == Direction.SHORT
    # Exit: BOT 10 @ 96 → covers for +40
    eng._account_for_fill(_fill("GLD", Direction.LONG, 10, 96.0))
    assert "GLD" not in eng._open_positions
    assert eng._risk.daily_pnl == 40.0, f"expected +40, got {eng._risk.daily_pnl}"
    print("PASS test_engine_short_round_trip")


def test_engine_partial_close() -> None:
    eng = TradingEngine(_cfg())
    eng._account_for_fill(_fill("AMZN", Direction.LONG, 10, 200.0))
    # Close half at +10/share
    eng._account_for_fill(_fill("AMZN", Direction.SHORT, 4, 210.0))
    assert eng._open_positions["AMZN"].quantity == 6
    assert eng._risk.daily_pnl == 40.0, f"expected +40, got {eng._risk.daily_pnl}"
    # Close the rest
    eng._account_for_fill(_fill("AMZN", Direction.SHORT, 6, 210.0))
    assert "AMZN" not in eng._open_positions
    assert eng._risk.daily_pnl == 100.0, f"expected +100, got {eng._risk.daily_pnl}"
    print("PASS test_engine_partial_close")


def test_engine_kill_switch_end_to_end() -> None:
    eng = TradingEngine(_cfg(max_daily_loss=500.0))
    eng._account_for_fill(_fill("NVDA", Direction.LONG, 100, 200.0))
    # Exit at a 7-point loss → -700, exceeds the 500 daily limit
    eng._account_for_fill(_fill("NVDA", Direction.SHORT, 100, 193.0))
    assert eng._risk.is_halted, "engine should be halted after breaching daily loss"
    assert eng._state["is_halted"] is True, "dashboard state should reflect the halt"

    # A fresh signal must now be rejected by the risk gate
    ok, reason = eng._risk.approve_signal(_signal("GLD", Direction.LONG), {})
    assert not ok, "no new entries allowed while halted"
    print("PASS test_engine_kill_switch_end_to_end")


class _RecordingBroker:
    """Stub broker that records flatten_symbol calls instead of trading."""

    def __init__(self) -> None:
        self.flatten_calls: list[dict] = []

    def flatten_symbol(self, symbol, strategy_id, quantity, position_side) -> None:
        self.flatten_calls.append({
            "symbol": symbol, "strategy_id": strategy_id,
            "quantity": quantity, "position_side": position_side,
        })


def _strategy_stub(symbol: str):
    """Minimal object satisfying the bits of BaseStrategy that _handle_signal uses."""
    class _S:
        strategy_id = "test"
        symbols = [symbol]
        def get_pending_levels(self, sym): return None
    return _S()


def test_flat_signal_closes_position() -> None:
    eng = TradingEngine(_cfg())
    eng._broker = _RecordingBroker()  # type: ignore[assignment]
    # Open a long via the ledger
    eng._account_for_fill(_fill("MSFT", Direction.LONG, 10, 100.0))
    # A FLAT signal should trigger a flatten of the open long
    eng._handle_signal(_signal("MSFT", Direction.FLAT), _strategy_stub("MSFT"))
    calls = eng._broker.flatten_calls  # type: ignore[attr-defined]
    assert len(calls) == 1, f"expected 1 flatten call, got {len(calls)}"
    assert calls[0]["symbol"] == "MSFT"
    assert calls[0]["quantity"] == 10
    assert calls[0]["position_side"] == Direction.LONG
    print("PASS test_flat_signal_closes_position")


def test_flat_signal_noop_when_flat() -> None:
    eng = TradingEngine(_cfg())
    eng._broker = _RecordingBroker()  # type: ignore[assignment]
    # No open position → FLAT should do nothing
    eng._handle_signal(_signal("MSFT", Direction.FLAT), _strategy_stub("MSFT"))
    assert eng._broker.flatten_calls == []  # type: ignore[attr-defined]
    print("PASS test_flat_signal_noop_when_flat")


def test_correlation_filter_blocks_when_concentrated() -> None:
    rm = RiskManager(_cfg())  # default: risk_on=(QQQ,MSFT,NVDA,AMZN), max=2
    # 2 risk-on positions already open → a new risk-on long is blocked
    ok, reason = rm.approve_signal(_signal("AMZN", Direction.LONG),
                                   {"MSFT": 10, "NVDA": 10})
    assert not ok and "correlation" in reason.lower(), reason
    print("PASS test_correlation_filter_blocks_when_concentrated")


def test_correlation_filter_allows_under_limit() -> None:
    rm = RiskManager(_cfg())
    # Only 1 risk-on open → under the limit → allowed
    ok, _ = rm.approve_signal(_signal("AMZN", Direction.LONG), {"MSFT": 10})
    assert ok
    print("PASS test_correlation_filter_allows_under_limit")


def test_correlation_filter_exempts_commodities() -> None:
    rm = RiskManager(_cfg())
    # GLD is not a risk-on symbol → not blocked even with risk-on slots full
    ok, _ = rm.approve_signal(_signal("GLD", Direction.LONG),
                              {"MSFT": 10, "NVDA": 10})
    assert ok, "commodities should be exempt from the risk-on correlation filter"
    print("PASS test_correlation_filter_exempts_commodities")


if __name__ == "__main__":
    tests = [
        test_record_close_updates_pnl_and_halts,
        test_short_pnl_sign,
        test_engine_long_round_trip,
        test_engine_short_round_trip,
        test_engine_partial_close,
        test_engine_kill_switch_end_to_end,
        test_flat_signal_closes_position,
        test_flat_signal_noop_when_flat,
        test_correlation_filter_blocks_when_concentrated,
        test_correlation_filter_allows_under_limit,
        test_correlation_filter_exempts_commodities,
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
