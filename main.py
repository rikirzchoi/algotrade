"""
main.py — application entry point.

NOTE (2026-06): the strategies wired below (momentum, bollinger, ORB, trend) were
rigorously backtested and found to have NO edge — see DECISIONS.md. The validated
approach is the 60/40 trend+equity ETF core (TRADING_RULES.md), which does not
need this engine. This live engine is retained for reference and for future
*satellite* strategies — do not deploy the rejected ones with real money.

Wires every component together in the correct order:

1. Load AppConfig (reads .env, applies defaults)
2. Instantiate TradingEngine (internally creates Broker, RiskManager, DatabaseWriter)
3. Register all active strategies with the engine
4. Install SIGINT / SIGTERM handlers for graceful shutdown
5. Print startup banner
6. Call engine.run() — blocks until shutdown

Usage::

    python main.py
"""

from __future__ import annotations

import logging
import signal
import sys
import threading

from config import AppConfig
from core.engine import TradingEngine
import uvicorn
from dashboard.server import create_server
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.bollinger_reversion import BollingerReversionStrategy
from strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    cfg = AppConfig.from_env()

    engine = TradingEngine(cfg)
    engine.register_strategy(MomentumBreakoutStrategy(cfg))
    engine.register_strategy(BollingerReversionStrategy(cfg))
    engine.register_strategy(OpeningRangeBreakoutStrategy(cfg))
    engine.register_strategy(TrendFollowingStrategy(cfg))

    def _shutdown(sig: int, frame: object) -> None:
        print("\nShutting down...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if "--no-dashboard" not in sys.argv:
        api = create_server(engine, cfg)
        api_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={"app": api, "host": "0.0.0.0", "port": 8000, "log_level": "warning"},
            daemon=True,
        )
        api_thread.start()
        print(f"API:       http://0.0.0.0:8000/api/docs")
        print(f"Dashboard: http://localhost:3000")

    mode = "PAPER TRADING" if cfg.is_paper_trading else "LIVE TRADING"
    print(f"\n{'='*50}")
    print(f"  AlgoTrade — {mode}")
    print(f"  Dashboard: http://{cfg.dashboard.host}:{cfg.dashboard.port}")
    print(f"  DB: {cfg.database.db_path}")
    print(
        f"  Risk: max loss=${cfg.risk.max_daily_loss_usd} "
        f"drawdown={cfg.risk.max_drawdown_pct:.0%}"
    )
    print(f"{'='*50}\n")

    engine.run()


if __name__ == "__main__":
    main()
