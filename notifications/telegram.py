"""
telegram.py — Telegram notification integration.

Sends real-time alerts and a daily 4 PM ET summary to a configured
Telegram bot. All sends are fire-and-forget; failures are logged but
never propagate to the calling thread.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, date
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import requests

if TYPE_CHECKING:
    from core.events import FillEvent

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Sends notifications to a Telegram chat via the Bot API."""

    def __init__(self, token: str, chat_id: str) -> None:
        self._url = _API.format(token=token)
        self._chat_id = chat_id
        self._last_summary: date | None = None
        self._engine_ref: Any = None
        self._summary_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public send helpers
    # ------------------------------------------------------------------

    def send(self, text: str) -> None:
        """Fire-and-forget message send. Never raises."""
        try:
            requests.post(
                self._url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)

    def notify_fill(self, fill: "FillEvent") -> None:
        side = "BUY" if fill.direction.value == "LONG" else "SELL"
        pnl_str = ""
        if fill.realised_pnl and fill.realised_pnl != 0:
            sign = "+" if fill.realised_pnl > 0 else ""
            pnl_str = f"\nPnL: <b>{sign}${fill.realised_pnl:.2f}</b>"
        text = (
            f"{'🟢' if side == 'BUY' else '🔴'} <b>{side} {fill.quantity} {fill.symbol}</b>\n"
            f"Strategy: {fill.strategy_id}\n"
            f"Price: ${fill.fill_price:.2f}"
            f"{pnl_str}"
        )
        self.send(text)

    def notify_kill_switch(self) -> None:
        self.send(
            "🚨 <b>KILL SWITCH ACTIVATED</b>\n"
            "All new orders are blocked. Check the dashboard."
        )

    def notify_connection_lost(self) -> None:
        self.send("⚠️ <b>Connection lost</b> — disconnected from TWS. Reconnecting...")

    def notify_connection_restored(self) -> None:
        self.send("✅ <b>Reconnected</b> — TWS connection restored.")

    def notify_data_stale(self, minutes: int) -> None:
        self.send(
            f"🟠 <b>MARKET DATA STALE</b>\n"
            f"No live bar in ~{minutes} min during trading hours. The engine is "
            f"connected but receiving no data — strategies are blind, no orders "
            f"will fire. Check TWS market-data subscription / data farm."
        )

    def notify_data_restored(self) -> None:
        self.send("✅ <b>Market data restored</b> — live bars are flowing again.")

    def notify_engine_start(self, paper: bool) -> None:
        mode = "PAPER" if paper else "LIVE"
        self.send(f"🚀 <b>AlgoTrade started</b> ({mode} trading)")

    # ------------------------------------------------------------------
    # Daily summary scheduler
    # ------------------------------------------------------------------

    def start_daily_summary(self, engine: Any) -> None:
        """Start background thread that sends a summary at 4 PM ET daily."""
        self._engine_ref = engine
        self._summary_thread = threading.Thread(
            target=self._summary_loop,
            daemon=True,
            name="telegram-summary",
        )
        self._summary_thread.start()

    def _summary_loop(self) -> None:
        while True:
            now = datetime.now(tz=ET)
            today = now.date()

            # Target: 4:05 PM ET (5 min after close to let fills settle)
            target = now.replace(hour=16, minute=5, second=0, microsecond=0)
            if now >= target or self._last_summary == today:
                # Already past 4:05 PM or already sent today — wait until tomorrow
                from datetime import timedelta
                tomorrow = (now + timedelta(days=1)).replace(
                    hour=16, minute=5, second=0, microsecond=0
                )
                sleep_secs = (tomorrow - now).total_seconds()
            else:
                sleep_secs = (target - now).total_seconds()

            time.sleep(max(sleep_secs, 1))

            if self._last_summary == datetime.now(tz=ET).date():
                continue

            self._send_daily_summary()
            self._last_summary = datetime.now(tz=ET).date()

    def _send_daily_summary(self) -> None:
        if self._engine_ref is None:
            return
        try:
            state = self._engine_ref.shared_state
            daily_pnl = state.get("daily_pnl", 0.0)
            drawdown = state.get("drawdown_pct", 0.0) * 100
            fills = state.get("fills_today", [])
            positions = state.get("positions", {})
            is_halted = state.get("is_halted", False)

            wins = sum(
                1 for f in fills
                if getattr(f, "realised_pnl", None) and f.realised_pnl > 0
            )
            closed = sum(
                1 for f in fills
                if getattr(f, "realised_pnl", None) and f.realised_pnl != 0
            )
            win_rate = f"{wins}/{closed}" if closed > 0 else "—"

            pnl_sign = "+" if daily_pnl >= 0 else ""
            emoji = "🟢" if daily_pnl >= 0 else "🔴"

            pos_lines = "\n".join(
                f"  {sym}: {qty:+d}" for sym, qty in positions.items()
            ) or "  None"

            halted_line = "\n⚠️ <b>Kill switch is active</b>" if is_halted else ""

            self.send(
                f"{emoji} <b>Daily Summary — {datetime.now(tz=ET).strftime('%b %d')}</b>\n"
                f"\nPnL today: <b>{pnl_sign}${daily_pnl:.2f}</b>"
                f"\nDrawdown:  {drawdown:.1f}%"
                f"\nFills:     {len(fills)}"
                f"\nWin/Loss:  {win_rate}"
                f"\n\nOpen positions:\n{pos_lines}"
                f"{halted_line}"
            )
        except Exception as exc:
            log.warning("Failed to send daily summary: %s", exc)
