"""
dashboard.server — FastAPI backend replacing the Dash app.

REST endpoints
--------------
GET  /api/state                      engine shared-state snapshot
GET  /api/strategies                 registered strategy ids
GET  /api/fills                      fills from SQLite
GET  /api/equity-curve               cumulative PnL time series
GET  /api/bars/{symbol}              OHLCV bars for a symbol
GET  /api/performance                per-strategy summary
GET  /api/system-events              recent system events
GET  /api/market-snapshot            last price + daily change per symbol
POST /api/strategies/{id}/toggle     enable / disable a strategy
POST /api/kill-switch                halt the engine
POST /api/resume                     clear the kill switch

WebSocket
---------
WS  /ws    streams full state payload to every connected client every second;
           clients receive an immediate snapshot on connect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import database.queries as queries
from config import AppConfig

logger = logging.getLogger(__name__)

_engine: Any = None
_config: AppConfig | None = None


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class _ConnectionManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
        logger.debug("WS client connected (%d total)", len(self._clients))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)
        logger.debug("WS client disconnected (%d remaining)", len(self._clients))

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_manager = _ConnectionManager()


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------

def _fill_to_dict(fill: Any) -> dict:
    direction = getattr(fill, "direction", None)
    direction_val = direction.value if hasattr(direction, "value") else str(direction)
    return {
        "strategy_id": getattr(fill, "strategy_id", ""),
        "symbol":      getattr(fill, "symbol", ""),
        "direction":   direction_val,
        "quantity":    getattr(fill, "quantity", 0),
        "fill_price":  float(getattr(fill, "fill_price", 0)),
        "commission":  float(getattr(fill, "commission", 0)),
        "timestamp":   str(getattr(fill, "timestamp", "")),
    }


def _state_payload() -> dict:
    if _engine is None:
        return {"connected": False}
    state = _engine.shared_state
    hb = state.get("last_heartbeat")
    return {
        "connected":         True,
        "is_running":        bool(state.get("is_running", False)),
        "is_halted":         bool(state.get("is_halted", False)),
        "daily_pnl":         float(state.get("daily_pnl", 0.0)),
        "peak_equity":       float(state.get("peak_equity", 0.0)),
        "drawdown_pct":      float(state.get("drawdown_pct", 0.0)),
        "positions":         state.get("positions", {}),
        "active_strategies": state.get("active_strategies", []),
        "fills_today":       [_fill_to_dict(f) for f in state.get("fills_today", [])],
        "error_count_today": int(state.get("error_count_today", 0)),
        "last_heartbeat":    str(hb) if hb else None,
    }


# ---------------------------------------------------------------------------
# Broadcaster background task
# ---------------------------------------------------------------------------

async def _broadcast_loop() -> None:
    while True:
        await asyncio.sleep(1)
        if _manager._clients:
            try:
                await _manager.broadcast(_state_payload())
            except Exception:
                logger.exception("broadcast_loop error")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_server(engine: Any, cfg: AppConfig) -> FastAPI:
    """Build and return the FastAPI application."""
    global _engine, _config
    _engine = engine
    _config = cfg

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(_broadcast_loop())
        yield
        task.cancel()

    app = FastAPI(title="AlgoTrade API", docs_url="/api/docs", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── REST ──────────────────────────────────────────────────────────

    @app.get("/api/state")
    async def get_state() -> dict:
        return _state_payload()

    @app.get("/api/strategies")
    async def get_strategies() -> list[str]:
        if _engine is None:
            return []
        return [s.strategy_id for s in _engine._strategies]

    @app.post("/api/strategies/{strategy_id}/toggle")
    async def toggle_strategy(strategy_id: str) -> dict:
        if _engine is None:
            return {"ok": False, "error": "engine not connected"}
        state = _engine.shared_state
        currently_active = strategy_id in set(state.get("active_strategies", []))
        _engine.toggle_strategy(strategy_id, not currently_active)
        return {"ok": True, "active": not currently_active}

    @app.post("/api/kill-switch")
    async def trigger_kill_switch() -> dict:
        if _engine is None:
            return {"ok": False}
        _engine.stop()
        return {"ok": True}

    @app.post("/api/resume")
    async def resume() -> dict:
        if _engine is None:
            return {"ok": False}
        _engine.risk_manager.resume()
        return {"ok": True}

    @app.get("/api/fills")
    async def get_fills(limit: int = 200) -> list[dict]:
        if _config is None:
            return []
        df = queries.get_fills(_config.database.db_path)
        if df.empty:
            return []
        df = df.tail(limit).copy()
        for col in df.select_dtypes(include="datetime64").columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    @app.get("/api/equity-curve")
    async def get_equity_curve() -> list[dict]:
        if _config is None:
            return []
        df = queries.get_equity_curve(_config.database.db_path)
        if df.empty:
            return []
        df = df.copy()
        df["exit_time"] = df["exit_time"].astype(str)
        return df.to_dict(orient="records")

    @app.get("/api/bars/{symbol}")
    async def get_bars(
        symbol: str,
        bar_size: str = "1 day",
        limit: int = 200,
    ) -> list[dict]:
        if _config is None:
            return []
        df = queries.get_recent_bars(_config.database.db_path, symbol, bar_size, limit)
        if df.empty:
            return []
        df = df.copy()
        for col in df.select_dtypes(include="datetime64").columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    @app.get("/api/performance")
    async def get_performance() -> list[dict]:
        if _config is None:
            return []
        df = queries.get_performance_summary(_config.database.db_path)
        if df.empty:
            return []
        return df.to_dict(orient="records")

    @app.get("/api/system-events")
    async def get_system_events(limit: int = 50) -> list[dict]:
        if _config is None:
            return []
        df = queries.get_system_events(_config.database.db_path, limit=limit)
        if df.empty:
            return []
        df = df.copy()
        for col in df.select_dtypes(include="datetime64").columns:
            df[col] = df[col].astype(str)
        return df.to_dict(orient="records")

    @app.get("/api/market-snapshot")
    async def get_market_snapshot() -> list[dict]:
        if _config is None:
            return []
        df = queries.get_last_bars_summary(_config.database.db_path)
        if df.empty:
            return []
        df = df.copy()
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(0)
        return df.to_dict(orient="records")

    # ── WebSocket ──────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await _manager.connect(ws)
        try:
            await ws.send_json(_state_payload())  # immediate snapshot on connect
            while True:
                await asyncio.sleep(30)           # keep-alive; broadcaster does the pushing
        except WebSocketDisconnect:
            _manager.disconnect(ws)
        except Exception:
            _manager.disconnect(ws)

    return app
