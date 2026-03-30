"""
database.queries — read-only aggregated queries for the Dash dashboard.

All functions open their own short-lived SQLite connection (read-only) and
return pandas DataFrames.  No writes happen here.

The dashboard calls these functions on a polling interval; they are designed
to be fast (< 50 ms) by relying on appropriate indexes created by
database.writer.

Available query groups
----------------------
Bars        — recent bars for one or all symbols (for charting)
Signals     — recent signals with strategy attribution
Fills       — trade history with realised P&L per round-trip
Performance — daily / cumulative P&L, win rate, Sharpe (rolling 20 days)
System      — recent system events and error log
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd


def _open_ro(db_path: Path) -> sqlite3.Connection | None:
    """Return a read-only connection, or None if the database file doesn't exist."""
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def get_recent_bars(
    db_path: Path,
    symbol: str,
    bar_size: str,
    limit: int = 200,
) -> pd.DataFrame:
    """Return the most recent *limit* bars for *symbol* at *bar_size*."""
    conn = _open_ro(db_path)
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            """SELECT id, symbol, timestamp, open, high, low, close, volume, bar_size, vwap
               FROM bars
               WHERE symbol = ? AND bar_size = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            conn,
            params=(symbol, bar_size, limit),
        )
        return df.iloc[::-1].reset_index(drop=True)
    except sqlite3.Error:
        return pd.DataFrame()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def get_recent_signals(
    db_path: Path,
    since: Optional[datetime] = None,
    strategy_id: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Return recent signals, optionally filtered by strategy and time window."""
    conn = _open_ro(db_path)
    if conn is None:
        return pd.DataFrame()
    try:
        clauses: list[str] = []
        params: list = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if strategy_id is not None:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        df = pd.read_sql_query(
            f"SELECT * FROM signals {where} ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=params,
        )
        return df.iloc[::-1].reset_index(drop=True)
    except sqlite3.Error:
        return pd.DataFrame()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fills / trade history
# ---------------------------------------------------------------------------

def get_fills(
    db_path: Path,
    since: Optional[datetime] = None,
    strategy_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return all fills, optionally filtered by strategy and start date."""
    conn = _open_ro(db_path)
    if conn is None:
        return pd.DataFrame()
    try:
        clauses: list[str] = []
        params: list = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since.isoformat())
        if strategy_id is not None:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df = pd.read_sql_query(
            f"SELECT * FROM fills {where} ORDER BY timestamp ASC",
            conn,
            params=params,
        )
        return df
    except sqlite3.Error:
        return pd.DataFrame()
    finally:
        conn.close()


def get_round_trips(
    db_path: Path,
    strategy_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return matched entry/exit pairs with per-trade P&L, duration, and MAE/MFE.

    Pairs are matched FIFO within each (strategy_id, symbol) group.
    """
    fills = get_fills(db_path, strategy_id=strategy_id)
    if fills.empty:
        return pd.DataFrame()

    round_trips: list[dict] = []

    for (strat, sym), group in fills.groupby(["strategy_id", "symbol"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        # Each open position: [remaining_qty, price, timestamp, remaining_commission]
        open_longs: list[list] = []
        open_shorts: list[list] = []

        for _, fill in group.iterrows():
            direction = fill["direction"]
            fill_qty = int(fill["quantity"])
            fill_price = float(fill["fill_price"])
            fill_commission = float(fill["commission"])
            fill_ts = fill["timestamp"]
            remaining_qty = fill_qty

            if direction == "LONG":
                # Close open SHORT positions FIFO, then open new LONG remainder
                while remaining_qty > 0 and open_shorts:
                    entry = open_shorts[0]
                    match_qty = min(remaining_qty, entry[0])
                    entry_comm = entry[3] * (match_qty / entry[0])
                    exit_comm = fill_commission * (match_qty / fill_qty)
                    pnl = (entry[1] - fill_price) * match_qty - entry_comm - exit_comm
                    round_trips.append({
                        "strategy_id": strat,
                        "symbol": sym,
                        "direction": "SHORT",
                        "entry_time": entry[2],
                        "exit_time": fill_ts,
                        "quantity": match_qty,
                        "entry_price": entry[1],
                        "exit_price": fill_price,
                        "pnl": pnl,
                        "commission": entry_comm + exit_comm,
                    })
                    remaining_qty -= match_qty
                    entry[0] -= match_qty
                    entry[3] -= entry_comm
                    if entry[0] <= 0:
                        open_shorts.pop(0)
                if remaining_qty > 0:
                    alloc_comm = fill_commission * (remaining_qty / fill_qty)
                    open_longs.append([remaining_qty, fill_price, fill_ts, alloc_comm])

            else:  # SHORT fill
                # Close open LONG positions FIFO, then open new SHORT remainder
                while remaining_qty > 0 and open_longs:
                    entry = open_longs[0]
                    match_qty = min(remaining_qty, entry[0])
                    entry_comm = entry[3] * (match_qty / entry[0])
                    exit_comm = fill_commission * (match_qty / fill_qty)
                    pnl = (fill_price - entry[1]) * match_qty - entry_comm - exit_comm
                    round_trips.append({
                        "strategy_id": strat,
                        "symbol": sym,
                        "direction": "LONG",
                        "entry_time": entry[2],
                        "exit_time": fill_ts,
                        "quantity": match_qty,
                        "entry_price": entry[1],
                        "exit_price": fill_price,
                        "pnl": pnl,
                        "commission": entry_comm + exit_comm,
                    })
                    remaining_qty -= match_qty
                    entry[0] -= match_qty
                    entry[3] -= entry_comm
                    if entry[0] <= 0:
                        open_longs.pop(0)
                if remaining_qty > 0:
                    alloc_comm = fill_commission * (remaining_qty / fill_qty)
                    open_shorts.append([remaining_qty, fill_price, fill_ts, alloc_comm])

    if not round_trips:
        return pd.DataFrame()

    df = pd.DataFrame(round_trips)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["duration"] = df["exit_time"] - df["entry_time"]
    return df.sort_values("exit_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def get_equity_curve(
    db_path: Path,
    strategy_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return a cumulative P&L time series (equity curve) from fills."""
    rt = get_round_trips(db_path, strategy_id=strategy_id)
    if rt.empty:
        return pd.DataFrame()
    rt = rt.sort_values("exit_time").copy()
    rt["cumulative_pnl"] = rt["pnl"].cumsum()
    return rt[["strategy_id", "symbol", "exit_time", "pnl", "cumulative_pnl"]].reset_index(drop=True)


def get_daily_pnl(
    db_path: Path,
    strategy_id: Optional[str] = None,
) -> pd.DataFrame:
    """Return realised P&L aggregated by date."""
    rt = get_round_trips(db_path, strategy_id=strategy_id)
    if rt.empty:
        return pd.DataFrame()
    rt = rt.copy()
    rt["exit_date"] = rt["exit_time"].dt.date
    daily = (
        rt.groupby(["strategy_id", "exit_date"])["pnl"]
        .sum()
        .reset_index()
        .rename(columns={"exit_date": "date"})
        .sort_values(["strategy_id", "date"])
        .reset_index(drop=True)
    )
    return daily


def get_performance_summary(db_path: Path) -> pd.DataFrame:
    """
    Return one row per strategy with: total P&L, win rate, trade count,
    Sharpe (rolling 20 trading days), and max drawdown.
    """
    rt = get_round_trips(db_path)
    if rt.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for strat_id, trades in rt.groupby("strategy_id"):
        trades = trades.copy().sort_values("exit_time")
        total_pnl = float(trades["pnl"].sum())
        trade_count = len(trades)
        win_rate = float((trades["pnl"] > 0).sum() / trade_count) if trade_count else 0.0

        # Sharpe over last 20 trading days of daily P&L
        trades["exit_date"] = trades["exit_time"].dt.date
        daily = trades.groupby("exit_date")["pnl"].sum().tail(20)
        sharpe = float(daily.mean() / daily.std() * (252 ** 0.5)) if len(daily) >= 2 else 0.0

        # Max drawdown from equity curve
        equity = trades["pnl"].cumsum()
        drawdown = float((equity - equity.cummax()).min())

        rows.append({
            "strategy_id": strat_id,
            "total_pnl": total_pnl,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# System events / health
# ---------------------------------------------------------------------------

def get_system_events(
    db_path: Path,
    limit: int = 50,
) -> pd.DataFrame:
    """Return the most recent system events (errors, warnings, lifecycle)."""
    conn = _open_ro(db_path)
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM system ORDER BY timestamp DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        return df.iloc[::-1].reset_index(drop=True)
    except sqlite3.Error:
        return pd.DataFrame()
    finally:
        conn.close()


def get_error_count_today(db_path: Path) -> int:
    """Return the number of ERROR-level system events recorded today."""
    conn = _open_ro(db_path)
    if conn is None:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM system WHERE kind = ? AND DATE(timestamp) = DATE('now')",
            ("ERROR",),
        ).fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
