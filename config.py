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
    stop_loss_pct: float = 0.02           # 2 % stop loss
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
class DashboardConfig:
    """Dash monitoring app settings."""

    host: str = "127.0.0.1"
    port: int = 8050
    live_update_interval_ms: int = 5_000
    perf_update_interval_ms: int = 60_000


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
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

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

        return cls(
            ibkr=ibkr,
            risk=risk,
            database=database,
            dashboard=dashboard,
        )
