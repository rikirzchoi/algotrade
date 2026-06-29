"""
test_broker_commissions — verifies the execDetails/commissionReport flow:
fills are buffered until the real commission arrives, then queued with it; a
missing report is flushed (commission=0) by the safety net; the IBKR float-max
sentinel is treated as zero.

Runnable standalone:

    .venv/bin/python tests/test_broker_commissions.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig
from core.broker import Broker
from core.events import Direction, FillEvent


def _broker() -> Broker:
    b = Broker(AppConfig())          # constructs only; does NOT connect
    b._strategy_order_map["7"] = "bollinger_reversion"
    return b


def _exec(exec_id: str, order_id: int = 7, side: str = "BOT",
          shares: int = 10, price: float = 100.0):
    return SimpleNamespace(
        execId=exec_id, orderId=order_id, side=side,
        shares=shares, price=price, time="20260623  10:30:00",
    )


def _contract(symbol: str = "NVDA"):
    return SimpleNamespace(symbol=symbol)


def _drain(b: Broker):
    out = []
    while True:
        try:
            out.append(b.event_queue.get_nowait())
        except Exception:
            break
    return out


def test_fill_buffered_until_commission() -> None:
    b = _broker()
    b.execDetails(1, _contract(), _exec("e1"))
    # Nothing queued yet — buffered awaiting commissionReport
    assert _drain(b) == [], "fill should be buffered, not queued, before commission"
    assert "e1" in b._pending_fills

    b.commissionReport(SimpleNamespace(execId="e1", commission=1.25, currency="USD"))
    events = _drain(b)
    assert len(events) == 1 and isinstance(events[0], FillEvent)
    assert events[0].commission == Decimal("1.25"), events[0].commission
    assert events[0].strategy_id == "bollinger_reversion"
    assert "e1" not in b._pending_fills
    print("PASS test_fill_buffered_until_commission")


def test_commission_sentinel_treated_as_zero() -> None:
    b = _broker()
    b.execDetails(1, _contract(), _exec("e2"))
    b.commissionReport(SimpleNamespace(execId="e2", commission=1.7976931348623157e308,
                                       currency="USD"))
    events = _drain(b)
    assert len(events) == 1
    assert events[0].commission == Decimal("0.0"), events[0].commission
    print("PASS test_commission_sentinel_treated_as_zero")


def test_unknown_commission_report_ignored() -> None:
    b = _broker()
    b.commissionReport(SimpleNamespace(execId="nope", commission=1.0, currency="USD"))
    assert _drain(b) == [], "commissionReport with no buffered fill must do nothing"
    print("PASS test_unknown_commission_report_ignored")


def test_stale_fill_flushed_with_zero_commission() -> None:
    b = _broker()
    b.execDetails(1, _contract(), _exec("e3"))
    assert _drain(b) == []
    b._flush_stale_pending_fills(max_age=0.0)   # force-flush immediately
    events = _drain(b)
    assert len(events) == 1
    assert events[0].commission == Decimal("0.0")
    assert "e3" not in b._pending_fills
    print("PASS test_stale_fill_flushed_with_zero_commission")


if __name__ == "__main__":
    tests = [
        test_fill_buffered_until_commission,
        test_commission_sentinel_treated_as_zero,
        test_unknown_commission_report_ignored,
        test_stale_fill_flushed_with_zero_commission,
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
