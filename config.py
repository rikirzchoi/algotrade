"""
config.py — single source of truth for all configuration and constants.

Every tunable parameter lives here.  No magic numbers anywhere else in the
codebase.  Values are exposed as frozen dataclasses so that accidental
mutation is prevented and IDE auto-complete works everywhere.

Environment variables (loaded from .env) override defaults for deployment-
sensitive settings such as the IB connection and capital parameters.

Config sections
---------------
IBKRConfig                — IB TWS / Gateway connection settings
InstrumentConfig          — tradeable universe grouped by strategy type
RiskConfig                — position limits, daily loss ceiling, sizing
MomentumBreakoutConfig    — N-day high breakout strategy parameters
BollingerReversionConfig  — Bollinger + RSI mean-reversion parameters
OpeningRangeBreakoutConfig — first-30-min ORB strategy parameters
DatabaseConfig            — SQLite path and async-writer settings
DashboardConfig           — Dash host, port, refresh intervals
AppConfig                 — root config that bundles all sub-configs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IBKRConfig:
    """Interactive Brokers TWS / Gateway connection parameters."""

    host: str = "127.0.0.1"
    port: int = 7497      # 7497 = paper trading, 7496 = live
    client_id: int = 1


# ---------------------------------------------------------------------------
# Tradeable universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentConfig:
    """Tradeable universe split by strategy type."""

    swing_symbols: tuple[str, ...] = ("QQQ",)
    intraday_symbols: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN")
    bollinger_symbols: tuple[str, ...] = ("MSFT", "NVDA", "AMZN")
    orb_symbols: tuple[str, ...] = ("AMZN",)
    commodity_symbols: tuple[str, ...] = ("GLD", "USO")


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """System-wide risk parameters enforced by core.risk.RiskManager."""

    capital: float = 91_000.0
    max_position_size: int = 100          # max shares per symbol
    max_daily_loss_usd: float = 500.0     # hard stop for the day in dollars
    max_drawdown_pct: float = 0.05        # 5 % from peak triggers kill switch
    risk_per_trade_pct: float = 0.01      # 1 % of capital risked per trade
    # Correlation / concentration filter: the set of symbols treated as
    # correlated "risk-on" equity exposure. A new LONG in any of these is
    # blocked once max_risk_on_positions of them are already open, preventing
    # piling into correlated risk. (Commodities like GLD/USO are excluded — they
    # diversify rather than add to risk-on exposure.)
    risk_on_symbols: tuple[str, ...] = ("QQQ", "MSFT", "NVDA", "AMZN")
    max_risk_on_positions: int = 2


# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MomentumBreakoutConfig:
    """Parameters for the N-day high momentum breakout strategy."""

    bar_size: str = "1 day"               # IBKR bar size string
    lookback_days: int = 10               # N-day high lookback
    volume_multiplier: float = 1.2        # volume must be 1.2x 20-day average
    profit_target_pct: float = 0.04       # 4 % profit target
    stop_loss_pct: float = 0.02           # fallback stop when ATR unavailable
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0      # stop = entry - 2 × ATR
    active: bool = True


@dataclass(frozen=True)
class BollingerReversionConfig:
    """Parameters for the Bollinger band + RSI mean-reversion strategy."""

    bar_size: str = "15 mins"
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    profit_target_pct: float = 0.015
    stop_loss_pct: float = 0.015
    active: bool = True
    flat_by: str = "15:45"                # flatten all positions by this time ET


@dataclass(frozen=True)
class OpeningRangeBreakoutConfig:
    """Parameters for the opening range breakout strategy."""

    bar_size: str = "5 mins"
    range_minutes: int = 30               # first 30 mins form the range
    profit_target_pct: float = 0.02
    stop_loss_pct: float = 0.01
    active: bool = True
    flat_by: str = "15:45"


@dataclass(frozen=True)
class TrendFollowingConfig:
    """Parameters for the 4-hour EMA-crossover trend-following strategy."""

    bar_size: str = "4 hours"
    fast_ema: int = 21                    # fast EMA period
    slow_ema: int = 55                    # slow EMA period
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0      # stop distance = multiplier × ATR
    atr_target_multiplier: float = 3.0    # target distance = multiplier × ATR
    active: bool = True


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatabaseConfig:
    """SQLite path and async-writer flush settings."""

    db_path: Path = Path("algotrade.db")
    flush_interval_seconds: float = 1.0


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelegramConfig:
    """Telegram bot notification settings."""

    token: str = ""
    chat_id: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)


@dataclass(frozen=True)
class DashboardConfig:
    """Dash monitoring app settings."""

    host: str = "127.0.0.1"
    port: int = 8050
    live_update_interval_ms: int = 5_000
    perf_update_interval_ms: int = 60_000


@dataclass(frozen=True)
class MarketDataConfig:
    """Live market-data watchdog settings.

    Guards against the silent failure where the engine stays connected and
    heartbeating but receives no live bars (e.g. a dead keepUpToDate stream or a
    missing real-time data subscription). During regular trading hours, if no
    live bar arrives for `staleness_timeout_seconds`, the engine raises a loud
    ERROR + alert (and optionally re-requests the subscriptions).
    """

    watchdog_enabled: bool = True
    staleness_timeout_seconds: int = 300        # 5 min with no live bar in RTH → alert
    resubscribe_on_stale: bool = True
    rth_start: tuple[int, int] = (9, 30)
    rth_end: tuple[int, int] = (16, 0)


# ---------------------------------------------------------------------------
# Root config — instantiated once in main.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration; bundles all sub-configs."""

    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    momentum_breakout: MomentumBreakoutConfig = field(default_factory=MomentumBreakoutConfig)
    bollinger_reversion: BollingerReversionConfig = field(default_factory=BollingerReversionConfig)
    opening_range_breakout: OpeningRangeBreakoutConfig = field(default_factory=OpeningRangeBreakoutConfig)
    trend_following: TrendFollowingConfig = field(default_factory=TrendFollowingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)

    @property
    def is_paper_trading(self) -> bool:
        """Return True when connected to the IB paper trading port (7497)."""
        return self.ibkr.port == 7497

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build an AppConfig with env-var overrides loaded from .env.

        Reads (if present):
            IB_HOST, IB_PORT, IB_CLIENT_ID
            CAPITAL, MAX_DAILY_LOSS, MAX_DRAWDOWN_PCT
            DB_PATH
            DASHBOARD_PORT
        """
        load_dotenv()

        ibkr = IBKRConfig(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=int(os.getenv("IB_PORT", "7497")),
            client_id=int(os.getenv("IB_CLIENT_ID", "1")),
        )
        risk = RiskConfig(
            capital=float(os.getenv("CAPITAL", "100000.0")),
            max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS", "500.0")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.05")),
        )
        database = DatabaseConfig(
            db_path=Path(os.getenv("DB_PATH", "algotrade.db")),
        )
        dashboard = DashboardConfig(
            port=int(os.getenv("DASHBOARD_PORT", "8050")),
        )

        telegram = TelegramConfig(
            token=os.getenv("TELEGRAM_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        return cls(
            ibkr=ibkr,
            risk=risk,
            database=database,
            dashboard=dashboard,
            telegram=telegram,
        )
