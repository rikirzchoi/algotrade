"""Tests for the market-data staleness watchdog (core.engine).

Validates the silent-data-blackout guard: during RTH, if no live bar arrives
within the timeout, the engine raises one ERROR + alert and re-subscribes; a
fresh live bar clears it. Run: .venv/bin/python tests/test_data_watchdog.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig, DatabaseConfig
from core.engine import TradingEngine
from core.events import SystemEventKind

ET = ZoneInfo("America/New_York")


class _FakeNotifier:
    def __init__(self): self.stale = 0; self.restored = 0
    def notify_data_stale(self, minutes): self.stale += 1
    def notify_data_restored(self): self.restored += 1


class _FakeDB:
    def __init__(self): self.events = []
    def write(self, e): self.events.append(e)


class _FakeBroker:
    def __init__(self): self.cancels = []; self.reqs = []
    def cancel_historical_data(self, rid): self.cancels.append(rid)
    def request_historical_data(self, rid, sym, bs, dur): self.reqs.append((rid, sym, bs, dur))


def _engine() -> TradingEngine:
    cfg = AppConfig(database=DatabaseConfig(db_path=Path("/tmp/_wd_test.db")))
    eng = TradingEngine(cfg)
    eng._db = _FakeDB()
    eng._notifier = _FakeNotifier()
    eng._broker = _FakeBroker()
    eng._subscriptions = [(1, "QQQ", "1 day", "1 Y")]
    eng._next_req_id = 2
    return eng


def test_is_rth():
    eng = _engine()
    assert eng._is_rth(datetime(2026, 6, 29, 10, 0, tzinfo=ET))         # Mon 10:00
    assert not eng._is_rth(datetime(2026, 6, 29, 9, 0, tzinfo=ET))      # before open
    assert not eng._is_rth(datetime(2026, 6, 29, 16, 30, tzinfo=ET))    # after close
    assert not eng._is_rth(datetime(2026, 6, 27, 11, 0, tzinfo=ET))     # Saturday
    print("ok _is_rth")


def test_stale_fires_once_and_resubscribes():
    eng = _engine()
    now_et = datetime(2026, 6, 29, 11, 0, tzinfo=ET)   # Monday, mid-session
    eng._prev_in_rth = True                            # already in session
    eng._last_realtime_bar_mono = 1000.0

    eng._check_data_staleness(now_et=now_et, now_mono=1000.0 + 299)    # within timeout
    assert not eng._data_stale and eng._notifier.stale == 0

    eng._check_data_staleness(now_et=now_et, now_mono=1000.0 + 301)    # past timeout
    assert eng._data_stale and eng._notifier.stale == 1
    assert eng._broker.cancels == [1] and len(eng._broker.reqs) == 1   # re-subscribed
    assert eng._broker.reqs[0][0] == 2                                 # fresh req_id
    assert any(getattr(e, "kind", None) == SystemEventKind.ERROR for e in eng._db.events)

    eng._check_data_staleness(now_et=now_et, now_mono=1000.0 + 900)    # still stale
    assert eng._notifier.stale == 1                                    # no second alert
    print("ok stale fires once + re-subscribes")


def test_recovery_clears_alert():
    eng = _engine()
    eng._data_stale = True
    eng._state["data_stale"] = True
    eng._note_realtime_bar()
    assert not eng._data_stale and eng._notifier.restored == 1
    assert eng._state["data_stale"] is False
    print("ok recovery clears")


def test_no_alert_outside_rth():
    eng = _engine()
    eng._prev_in_rth = False
    eng._last_realtime_bar_mono = 0.0
    eng._check_data_staleness(now_et=datetime(2026, 6, 29, 3, 0, tzinfo=ET),
                              now_mono=10_000.0)        # 3am, way past timeout
    assert not eng._data_stale and eng._notifier.stale == 0
    print("ok no alert outside RTH")


def test_open_transition_rearms():
    eng = _engine()
    eng._prev_in_rth = False
    eng._data_stale = True                              # stale from a prior state
    # First RTH tick of the session → clock restarts, alert cleared, no immediate fire
    eng._check_data_staleness(now_et=datetime(2026, 6, 29, 9, 30, tzinfo=ET),
                              now_mono=5_000.0)
    assert not eng._data_stale and eng._notifier.stale == 0
    print("ok open transition re-arms")


if __name__ == "__main__":
    test_is_rth()
    test_stale_fires_once_and_resubscribes()
    test_recovery_clears_alert()
    test_no_alert_outside_rth()
    test_open_transition_rearms()
    print("\nall data-watchdog tests passed")
