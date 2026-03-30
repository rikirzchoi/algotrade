"""
dashboard.app — three-tab Dash monitoring interface.

Tab 1 — Live Operations
    Real-time equity curve, open positions, fills, strategy toggles,
    kill-switch button with confirmation modal.

Tab 2 — Performance
    Cumulative P&L, drawdown, per-strategy breakdown, monthly heatmap,
    win/loss histogram.

Tab 3 — Health
    System events log, signal frequency, commission tracker, macro
    regime panel, upcoming events.

Data is read from SQLite via database.queries; live engine state is read
from engine.shared_state.  The module exposes create_app() and
run_dashboard() for use from main.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html

import database.queries as queries
from config import AppConfig

logger = logging.getLogger(__name__)

# Module-level references set by create_app() — read by callbacks at runtime.
engine = None   # TradingEngine | None
config: Optional[AppConfig] = None


# ---------------------------------------------------------------------------
# Figure / table helpers
# ---------------------------------------------------------------------------

def _empty_figure(message: str = "No data yet — waiting for first trade") -> go.Figure:
    """Empty Plotly figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888888"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7280"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def _dark_layout(**kwargs) -> dict:
    """Base dark layout dict merged into every chart."""
    base: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6b7280"),
        xaxis=dict(gridcolor="#1e1e2a", zeroline=False),
        yaxis=dict(gridcolor="#1e1e2a", zeroline=False),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    base.update(kwargs)
    return base


def _empty_table_row(message: str, cols: int) -> html.Tr:
    return html.Tr(html.Td(
        message,
        colSpan=cols,
        style={"textAlign": "center", "color": "#6b7280", "fontStyle": "italic"},
    ))


def _pnl_color(value: float) -> str:
    return "#00e676" if value >= 0 else "#ff5252"


# ---------------------------------------------------------------------------
# Tab 1 — Live layout
# ---------------------------------------------------------------------------

def _live_tab(strategy_ids: list[str] | None = None) -> dbc.Tab:
    return dbc.Tab(
        label="Live",
        tab_id="tab-live",
        children=[
            dcc.Interval(id="live-interval", interval=5000, n_intervals=0),

            # Row 1 — Status cards
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("DAILY PNL", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="status-daily-pnl", children="$0.00", className="mb-0",
                            style={"fontSize": "28px", "fontWeight": "600"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }, className="h-100"), width=3),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("DRAWDOWN", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="status-drawdown", children="0.00%", className="mb-0",
                            style={"fontSize": "28px", "fontWeight": "600"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }, className="h-100"), width=3),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("CONNECTION", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="status-connection", children="⬤ Unknown", className="mb-0",
                            style={"fontSize": "28px", "fontWeight": "600"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }, className="h-100"), width=3),

                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("KILL SWITCH", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.Div(id="status-halted", children=html.Span(
                        "Active", style={"color": "#00e676"}
                    ), style={"fontSize": "28px", "fontWeight": "600"}),
                    # Persistent Resume button — shown only when halted
                    dbc.Button(
                        "Resume",
                        id="resume-btn",
                        size="sm",
                        className="mt-1",
                        style={
                            "display": "none",
                            "background": "#f5a623", "color": "#000",
                            "border": "none", "borderRadius": "8px",
                            "fontWeight": "600", "padding": "6px 16px",
                        },
                        n_clicks=0,
                    ),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }, className="h-100"), width=3),
            ], className="mb-3 mt-3", style={"gap": "0", "--bs-gutter-x": "12px"}),

            # Row 2 — Live equity curve
            dbc.Row([
                dbc.Col(html.Div(
                    dcc.Graph(id="live-equity-chart", figure=_empty_figure()),
                    style={
                        "background": "#1a1a24", "border": "1px solid #2a2a3a",
                        "borderRadius": "12px", "padding": "8px",
                    },
                ), width=12),
            ], className="mb-3"),

            # Row 3 — Fills + Positions
            dbc.Row([
                dbc.Col([
                    html.H5("Today's Fills", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="fills-table"),
                ], width=7),
                dbc.Col([
                    html.H5("Open Positions", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="positions-table"),
                ], width=5),
            ], className="mb-3"),

            # Row 4 — Strategy controls (static cards; badges/buttons updated by interval)
            dbc.Row([
                dbc.Col([
                    html.H5("Strategy Controls", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(
                        _build_static_strategy_controls(strategy_ids or []),
                        id="strategy-controls",
                    ),
                    # Hidden output for toggle callbacks
                    html.Div(id="strategy-toggle-output", style={"display": "none"}),
                ], width=12),
            ], className="mb-3"),

            # Row 5 — Kill switch
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "EMERGENCY STOP",
                        id="kill-switch-btn",
                        className="w-100 kill-switch-btn",
                        n_clicks=0,
                        style={
                            "background": "transparent",
                            "border": "2px solid #ff5252",
                            "color": "#ff5252",
                            "borderRadius": "8px",
                            "fontWeight": "700",
                            "fontSize": "14px",
                            "letterSpacing": "0.1em",
                        },
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Confirm Emergency Stop")),
                        dbc.ModalBody(
                            "This will immediately halt all trading and cancel all open orders. "
                            "Are you sure?"
                        ),
                        dbc.ModalFooter([
                            dbc.Button("Cancel", id="kill-switch-cancel",
                                       color="secondary", className="me-2", n_clicks=0),
                            dbc.Button("CONFIRM STOP", id="kill-switch-confirm",
                                       color="danger", n_clicks=0),
                        ]),
                    ], id="kill-switch-modal", is_open=False),
                ], width=12),
            ], className="mb-3"),
        ],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Performance layout
# ---------------------------------------------------------------------------

def _performance_tab() -> dbc.Tab:
    return dbc.Tab(
        label="Performance",
        tab_id="tab-performance",
        children=[
            dcc.Interval(id="perf-interval", interval=60_000, n_intervals=0),

            # Row 1 — Summary cards
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("TOTAL PNL", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="perf-total-pnl", children="$0.00",
                            style={"fontSize": "28px", "fontWeight": "600"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("SHARPE RATIO", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="perf-sharpe", children="0.00",
                            style={"fontSize": "28px", "fontWeight": "600", "color": "#e8e8f0"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("MAX DRAWDOWN", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="perf-max-dd", children="$0.00",
                            style={"fontSize": "28px", "fontWeight": "600", "color": "#ff5252"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P("WIN RATE", style={
                        "fontSize": "12px", "textTransform": "uppercase",
                        "letterSpacing": "0.08em", "color": "#6b7280", "marginBottom": "8px",
                    }),
                    html.H3(id="perf-win-rate", children="0.0%",
                            style={"fontSize": "28px", "fontWeight": "600", "color": "#e8e8f0"}),
                ]), style={
                    "background": "#1a1a24", "border": "1px solid #2a2a3a",
                    "borderRadius": "12px", "borderTop": "2px solid #f5a623",
                    "padding": "20px 24px",
                }), width=3),
            ], className="mb-3 mt-3", style={"--bs-gutter-x": "12px"}),

            # Row 2 — Equity + Drawdown
            dbc.Row([
                dbc.Col(html.Div(
                    dcc.Graph(id="perf-equity-chart", figure=_empty_figure()),
                    style={
                        "background": "#1a1a24", "border": "1px solid #2a2a3a",
                        "borderRadius": "12px", "padding": "8px",
                    },
                ), width=6),
                dbc.Col(html.Div(
                    dcc.Graph(id="perf-drawdown-chart", figure=_empty_figure()),
                    style={
                        "background": "#1a1a24", "border": "1px solid #2a2a3a",
                        "borderRadius": "12px", "padding": "8px",
                    },
                ), width=6),
            ], className="mb-3"),

            # Row 3 — Per-strategy table
            dbc.Row([
                dbc.Col([
                    html.H5("Per-Strategy Breakdown", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="perf-summary-table"),
                ], width=12),
            ], className="mb-3"),

            # Row 4 — Monthly heatmap
            dbc.Row([
                dbc.Col([
                    html.H5("Monthly PnL Heatmap", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(
                        dcc.Graph(id="monthly-heatmap", figure=_empty_figure()),
                        style={
                            "background": "#1a1a24", "border": "1px solid #2a2a3a",
                            "borderRadius": "12px", "padding": "8px",
                        },
                    ),
                ], width=12),
            ], className="mb-3"),

            # Row 5 — Win/loss histogram
            dbc.Row([
                dbc.Col([
                    html.H5("Win / Loss Distribution", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(
                        dcc.Graph(id="pnl-histogram", figure=_empty_figure()),
                        style={
                            "background": "#1a1a24", "border": "1px solid #2a2a3a",
                            "borderRadius": "12px", "padding": "8px",
                        },
                    ),
                ], width=12),
            ], className="mb-3"),
        ],
    )


# ---------------------------------------------------------------------------
# Tab 3 — Health layout
# ---------------------------------------------------------------------------

def _health_tab() -> dbc.Tab:
    return dbc.Tab(
        label="Health",
        tab_id="tab-health",
        children=[
            # Row 1 — System events
            dbc.Row([
                dbc.Col([
                    html.H5("System Events", className="mb-2 mt-3",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="system-events-table"),
                ], width=12),
            ], className="mb-3"),

            # Row 2 — Signal frequency + Commission tracker
            dbc.Row([
                dbc.Col([
                    html.H5("Signal Frequency", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(
                        dcc.Graph(id="signal-freq-chart", figure=_empty_figure()),
                        style={
                            "background": "#1a1a24", "border": "1px solid #2a2a3a",
                            "borderRadius": "12px", "padding": "8px",
                        },
                    ),
                ], width=6),
                dbc.Col([
                    html.H5("Avg Commission per Trade", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(
                        dcc.Graph(id="slippage-chart", figure=_empty_figure()),
                        style={
                            "background": "#1a1a24", "border": "1px solid #2a2a3a",
                            "borderRadius": "12px", "padding": "8px",
                        },
                    ),
                ], width=6),
            ], className="mb-3"),

            # Row 3 — Macro regime
            dbc.Row([
                dbc.Col([
                    html.H5("Macro Regime", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="macro-regime-panel"),
                ], width=12),
            ], className="mb-3"),

            # Row 4 — Upcoming events
            dbc.Row([
                dbc.Col([
                    html.H5("Upcoming Events", className="mb-2",
                            style={"color": "#e8e8f0", "fontWeight": "500"}),
                    html.Div(id="upcoming-events-table"),
                ], width=12),
            ], className="mb-3"),
        ],
    )


# ---------------------------------------------------------------------------
# Table builders (shared across callbacks)
# ---------------------------------------------------------------------------

_BADGE_LONG = {
    "backgroundColor": "rgba(0,230,118,0.15)", "color": "#00e676",
    "border": "1px solid #00e676", "borderRadius": "99px",
    "padding": "2px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BADGE_SHORT = {
    "backgroundColor": "rgba(255,82,82,0.15)", "color": "#ff5252",
    "border": "1px solid #ff5252", "borderRadius": "99px",
    "padding": "2px 10px", "fontSize": "11px", "fontWeight": "600",
}


def _build_positions_table(positions: dict) -> dbc.Table:
    headers = [html.Th("Symbol"), html.Th("Quantity"), html.Th("Side")]
    if not positions:
        return dbc.Table([
            html.Thead(html.Tr(headers)),
            html.Tbody([_empty_table_row("No open positions", 3)]),
        ], bordered=True, dark=True, hover=True, size="sm")

    rows = [
        html.Tr([
            html.Td(symbol),
            html.Td(abs(qty)),
            html.Td(html.Span(
                "LONG" if qty > 0 else "SHORT",
                style=_BADGE_LONG if qty > 0 else _BADGE_SHORT,
            )),
        ])
        for symbol, qty in positions.items()
    ]
    return dbc.Table([
        html.Thead(html.Tr(headers)),
        html.Tbody(rows),
    ], bordered=True, dark=True, hover=True, size="sm")


def _build_fills_table(fills_df: pd.DataFrame) -> dbc.Table:
    col_headers = ["Time", "Strategy", "Symbol", "Side", "Qty", "Price"]
    thead = html.Thead(html.Tr([html.Th(h) for h in col_headers]))

    if fills_df.empty:
        return dbc.Table([
            thead,
            html.Tbody([_empty_table_row("No fills today", len(col_headers))]),
        ], bordered=True, dark=True, hover=True, size="sm")

    df = fills_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ascending=False)

    rows = []
    for _, row in df.iterrows():
        direction = str(row.get("direction", ""))
        side_badge_style = _BADGE_LONG if direction == "LONG" else _BADGE_SHORT
        ts = row["timestamp"]
        time_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        price = row.get("fill_price")
        price_str = f"${float(price):.2f}" if price is not None else ""
        rows.append(html.Tr([
            html.Td(time_str),
            html.Td(str(row.get("strategy_id", ""))),
            html.Td(str(row.get("symbol", ""))),
            html.Td(html.Span(direction, style=side_badge_style)),
            html.Td(str(row.get("quantity", ""))),
            html.Td(price_str),
        ]))

    return dbc.Table([
        thead,
        html.Tbody(rows),
    ], bordered=True, dark=True, hover=True, size="sm", responsive=True)


_BADGE_ACTIVE_STYLE = {
    "backgroundColor": "rgba(0,230,118,0.2)", "color": "#00e676",
    "border": "1px solid #00e676", "borderRadius": "99px",
    "padding": "3px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BADGE_INACTIVE_STYLE = {
    "backgroundColor": "rgba(255,82,82,0.2)", "color": "#ff5252",
    "border": "1px solid #ff5252", "borderRadius": "99px",
    "padding": "3px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BTN_TOGGLE_STYLE = {
    "background": "#f5a623", "color": "#000",
    "border": "none", "borderRadius": "8px",
    "fontWeight": "600", "padding": "6px 16px",
    "fontSize": "12px",
}
_CARD_STRATEGY_STYLE = {
    "background": "#1a1a24", "border": "1px solid #2a2a3a",
    "borderRadius": "12px",
}


def _build_static_strategy_controls(strategy_ids: list[str]) -> list:
    """Build strategy control cards with stable component IDs (rendered once at startup)."""
    if not strategy_ids:
        return [html.P("No strategies registered.", className="text-muted")]
    cards = []
    for sid in strategy_ids:
        cards.append(dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Strong(sid, style={"color": "#e8e8f0", "fontWeight": "500",
                                            "fontSize": "14px"}), width=4),
            dbc.Col(dbc.Badge("Active", id=f"badge-{sid}", color="success",
                              style=_BADGE_ACTIVE_STYLE), width=2),
            dbc.Col(dbc.Button(
                "Pause",
                id=f"toggle-{sid}",
                size="sm",
                n_clicks=0,
                style=_BTN_TOGGLE_STYLE,
            ), width=2),
            dbc.Col(html.Small("Fills today: 0", id=f"fills-{sid}",
                               style={"color": "#6b7280"}), width=4),
        ], align="center")), className="mb-2", style=_CARD_STRATEGY_STYLE))
    return cards


def _build_strategy_controls(state: dict, strategy_ids: list[str]) -> list:
    active_set = set(state.get("active_strategies", []))
    fills_today = state.get("fills_today", [])

    cards = []
    for sid in strategy_ids:
        is_active = sid in active_set
        fill_count = sum(
            1 for f in fills_today
            if getattr(f, "strategy_id", None) == sid
        )
        cards.append(dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Strong(sid, style={"color": "#e8e8f0", "fontWeight": "500",
                                            "fontSize": "14px"}), width=4),
            dbc.Col(dbc.Badge(
                "Active" if is_active else "Inactive",
                color="success" if is_active else "danger",
                style=_BADGE_ACTIVE_STYLE if is_active else _BADGE_INACTIVE_STYLE,
            ), width=2),
            dbc.Col(dbc.Button(
                "Pause" if is_active else "Resume",
                id=f"toggle-{sid}",
                size="sm",
                n_clicks=0,
                style=_BTN_TOGGLE_STYLE,
            ), width=2),
            dbc.Col(html.Small(f"Fills today: {fill_count}",
                               style={"color": "#6b7280"}), width=4),
        ], align="center")), className="mb-2", style=_CARD_STRATEGY_STYLE))

    return cards


# ---------------------------------------------------------------------------
# Macro regime panel builders
# ---------------------------------------------------------------------------

def _build_regime_panel() -> html.Div:
    try:
        from macro.regime import load_regime
        regime = load_regime(Path("macro/regime.json"))
    except Exception:
        regime = None

    if regime is None:
        return dbc.Alert(
            "Regime file not found or unreadable. "
            "Create macro/regime.json to enable this panel.",
            color="secondary",
        )

    # Determine risk-appetite display from whatever field is available
    state_val = getattr(regime, "state", None)
    risk_appetite = getattr(regime, "risk_appetite", None)

    if state_val is not None:
        name = state_val.value if hasattr(state_val, "value") else str(state_val)
        if name == "RISK_ON":
            ra_color, ra_label = "success", "High / Risk On"
        elif name in ("RISK_OFF", "HALTED"):
            ra_color, ra_label = "danger", name.replace("_", " ").title()
        else:
            ra_color, ra_label = "warning", "Neutral"
    elif risk_appetite is not None:
        rv = str(risk_appetite).lower()
        ra_color = "success" if rv == "high" else ("danger" if rv == "low" else "warning")
        ra_label = str(risk_appetite)
    else:
        ra_color, ra_label = "secondary", "Unknown"

    notes = getattr(regime, "notes", "")
    updated = getattr(regime, "updated_at", getattr(regime, "last_updated", "Unknown"))
    trend = getattr(regime, "trend_regime", None)
    fed = getattr(regime, "fed_stance", None)
    geo = getattr(regime, "geopolitical_tension", None)
    sector_bias = getattr(regime, "sector_bias", [])

    info_rows: list = [
        dbc.Row([
            dbc.Col(html.Strong("Risk Appetite:"), width=3),
            dbc.Col(dbc.Badge(ra_label, color=ra_color), width=9),
        ], className="mb-2"),
    ]
    if trend is not None:
        info_rows.append(dbc.Row([
            dbc.Col(html.Strong("Trend Regime:"), width=3),
            dbc.Col(str(trend), width=9),
        ], className="mb-2"))
    if fed is not None:
        info_rows.append(dbc.Row([
            dbc.Col(html.Strong("Fed Stance:"), width=3),
            dbc.Col(str(fed), width=9),
        ], className="mb-2"))
    if geo is not None:
        info_rows.append(dbc.Row([
            dbc.Col(html.Strong("Geopolitical Tension:"), width=3),
            dbc.Col(str(geo), width=9),
        ], className="mb-2"))
    if sector_bias:
        badges = (
            [dbc.Badge(s, color="info", className="me-1") for s in sector_bias]
            if isinstance(sector_bias, list)
            else [dbc.Badge(str(sector_bias), color="info")]
        )
        info_rows.append(dbc.Row([
            dbc.Col(html.Strong("Sector Bias:"), width=3),
            dbc.Col(badges, width=9),
        ], className="mb-2"))
    if notes:
        info_rows.append(dbc.Row([
            dbc.Col(html.Strong("Notes:"), width=3),
            dbc.Col(html.Em(notes), width=9),
        ], className="mb-2"))
    info_rows.append(dbc.Row([
        dbc.Col(html.Strong("Last Updated:"), width=3),
        dbc.Col(html.Small(str(updated), className="text-muted"), width=9),
    ], className="mb-2"))

    return dbc.Card(dbc.CardBody([
        *info_rows,
        dbc.Button(
            "Edit in regime.json",
            disabled=True,
            color="secondary",
            size="sm",
            title="Edit macro/regime.json and restart dashboard",
            className="mt-2",
        ),
    ]))


def _build_upcoming_events_table() -> dbc.Table:
    try:
        from macro.regime import load_regime
        regime = load_regime(Path("macro/regime.json"))
        upcoming = getattr(regime, "upcoming_events", None) if regime else None
    except Exception:
        upcoming = None

    headers = ["Date", "Event", "Impact"]
    thead = html.Thead(html.Tr([html.Th(h) for h in headers]))

    if not upcoming:
        return dbc.Table([
            thead,
            html.Tbody([_empty_table_row(
                "No upcoming events in regime.json", len(headers)
            )]),
        ], bordered=True, dark=True, size="sm")

    try:
        upcoming = sorted(
            upcoming,
            key=lambda e: e.get("date", "") if isinstance(e, dict) else "",
        )
    except Exception:
        pass

    rows = []
    for event in upcoming:
        if not isinstance(event, dict):
            continue
        impact = str(event.get("impact", "")).lower()
        impact_color = (
            "#ff5252" if impact == "high"
            else ("#f5a623" if impact == "medium" else "#00e676")
        )
        rows.append(html.Tr([
            html.Td(str(event.get("date", ""))),
            html.Td(str(event.get("event", ""))),
            html.Td(str(event.get("impact", "")), style={"color": impact_color}),
        ]))

    if not rows:
        return dbc.Table([
            thead,
            html.Tbody([_empty_table_row(
                "No upcoming events in regime.json", len(headers)
            )]),
        ], bordered=True, dark=True, size="sm")

    return dbc.Table([thead, html.Tbody(rows)],
                     bordered=True, dark=True, size="sm")


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def _register_callbacks(
    app: dash.Dash,
    cfg: AppConfig,
    strategy_ids: list[str],
) -> None:
    """Register all Dash callbacks for live data refresh and interactivity."""

    # ------------------------------------------------------------------
    # Live tab
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("live-equity-chart", "figure"),
            Output("status-daily-pnl", "children"),
            Output("status-drawdown", "children"),
            Output("status-connection", "children"),
            Output("status-halted", "children"),
            Output("resume-btn", "style"),
            Output("positions-table", "children"),
            Output("fills-table", "children"),
        ],
        Input("live-interval", "n_intervals"),
    )
    def update_live_tab(n: int):  # noqa: ANN001
        db_path = cfg.database.db_path
        state = engine.shared_state if engine is not None else {}

        # Daily PnL
        daily_pnl: float = float(state.get("daily_pnl", 0.0))
        pnl_text = html.Span(
            f"${daily_pnl:+,.2f}",
            style={"color": _pnl_color(daily_pnl)},
        )

        # Drawdown
        dd_pct: float = float(state.get("drawdown_pct", 0.0))
        dd_val = dd_pct * 100
        dd_color = (
            "#ff5252" if dd_val > 4
            else ("#f5a623" if dd_val > 2 else "#e8e8f0")
        )
        dd_text = html.Span(f"{dd_val:.2f}%", style={"color": dd_color})

        # Connection status — heartbeat within 60 s = connected
        last_hb = state.get("last_heartbeat")
        connected = False
        if last_hb is not None:
            try:
                from zoneinfo import ZoneInfo
                hb_dt = pd.to_datetime(last_hb)
                if hb_dt.tzinfo is None:
                    hb_dt = hb_dt.replace(tzinfo=ZoneInfo("America/New_York"))
                now_et = datetime.now(tz=ZoneInfo("America/New_York"))
                connected = (now_et - hb_dt).total_seconds() < 60
            except Exception:
                connected = False
        conn_text = html.Span(
            "⬤ Connected" if connected else "⬤ Disconnected",
            className="connection-dot-connected" if connected else "connection-dot-disconnected",
            style={"color": "#00e676" if connected else "#ff5252"},
        )

        # Kill-switch / halted
        is_halted: bool = bool(state.get("is_halted", False))
        if is_halted:
            halted_content = html.Span("HALTED",
                                       style={"color": "#ff5252", "fontWeight": "bold"})
            resume_style = {
                "display": "inline-block",
                "background": "#f5a623", "color": "#000",
                "border": "none", "borderRadius": "8px",
                "fontWeight": "600", "padding": "6px 16px",
            }
        else:
            halted_content = html.Span("Active", style={"color": "#00e676"})
            resume_style = {"display": "none"}

        # Equity curve — today's trades
        today_start = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            eq_df = queries.get_equity_curve(db_path)
            if not eq_df.empty:
                eq_df["exit_time"] = pd.to_datetime(eq_df["exit_time"])
                today_eq = eq_df[eq_df["exit_time"] >= today_start]
                if not today_eq.empty:
                    fig = go.Figure()
                    colors = ["#f5a623", "#4488ff", "#00e676", "#ff5252", "#aa44ff"]
                    for i, (sid, grp) in enumerate(today_eq.groupby("strategy_id")):
                        grp = grp.sort_values("exit_time")
                        fig.add_trace(go.Scatter(
                            x=grp["exit_time"],
                            y=grp["cumulative_pnl"],
                            mode="lines",
                            name=str(sid),
                            line=dict(color=colors[i % len(colors)], width=2),
                        ))
                    fig.update_layout(_dark_layout(title="Today's Equity Curve"))
                else:
                    fig = _empty_figure()
            else:
                fig = _empty_figure()
        except Exception:
            fig = _empty_figure()

        # Today's fills
        try:
            fills_df = queries.get_fills(db_path, since=today_start)
        except Exception:
            fills_df = pd.DataFrame()
        fills_tbl = _build_fills_table(fills_df)

        # Open positions
        positions = state.get("positions", {})
        pos_tbl = _build_positions_table(positions)

        return (
            fig, pnl_text, dd_text, conn_text,
            halted_content, resume_style,
            pos_tbl, fills_tbl,
        )

    # ------------------------------------------------------------------
    # Resume (clears risk-manager kill switch)
    # ------------------------------------------------------------------

    @app.callback(
        Output("resume-btn", "n_clicks"),
        Input("resume-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_resume(n_clicks: int) -> int:
        if n_clicks and engine is not None:
            try:
                engine.risk_manager.resume()
                logger.info("Dashboard: kill switch cleared via Resume button")
            except Exception as exc:
                logger.error("Dashboard resume error: %s", exc)
        return 0

    # ------------------------------------------------------------------
    # Strategy badge / button-label refresh (interval-driven, read-only)
    # ------------------------------------------------------------------

    if strategy_ids:
        badge_outputs = []
        for sid in strategy_ids:
            badge_outputs += [
                Output(f"badge-{sid}", "children"),
                Output(f"badge-{sid}", "color"),
                Output(f"toggle-{sid}", "children"),
                Output(f"toggle-{sid}", "color"),
                Output(f"fills-{sid}", "children"),
            ]

        @app.callback(
            badge_outputs,
            Input("live-interval", "n_intervals"),
        )
        def update_strategy_badges(n: int):  # noqa: ANN001
            state = engine.shared_state if engine is not None else {}
            active_set = set(state.get("active_strategies", []))
            fills_today = state.get("fills_today", [])
            results = []
            for sid in strategy_ids:
                is_active = sid in active_set
                fill_count = sum(
                    1 for f in fills_today
                    if getattr(f, "strategy_id", None) == sid
                )
                results += [
                    "Active" if is_active else "Inactive",
                    "success" if is_active else "danger",
                    "Pause" if is_active else "Resume",
                    "warning" if is_active else "success",
                    f"Fills today: {fill_count}",
                ]
            return results

    # ------------------------------------------------------------------
    # Strategy toggles (one input per known strategy)
    # ------------------------------------------------------------------

    if strategy_ids:
        @app.callback(
            Output("strategy-toggle-output", "children"),
            [Input(f"toggle-{sid}", "n_clicks") for sid in strategy_ids],
            prevent_initial_call=True,
        )
        def toggle_strategy(*_args):  # noqa: ANN001
            ctx = callback_context
            if not ctx.triggered or engine is None:
                return ""
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            sid = button_id.removeprefix("toggle-")
            state = engine.shared_state
            currently_active = sid in set(state.get("active_strategies", []))
            engine.toggle_strategy(sid, not currently_active)
            action = "deactivated" if currently_active else "activated"
            logger.info("Dashboard toggled strategy %s → %s", sid, action)
            return f"{sid} {action}"

    # ------------------------------------------------------------------
    # Kill-switch modal
    # ------------------------------------------------------------------

    @app.callback(
        Output("kill-switch-modal", "is_open"),
        [
            Input("kill-switch-btn", "n_clicks"),
            Input("kill-switch-confirm", "n_clicks"),
            Input("kill-switch-cancel", "n_clicks"),
        ],
        State("kill-switch-modal", "is_open"),
        prevent_initial_call=True,
    )
    def handle_kill_switch(
        open_clicks: int,
        confirm: int,
        cancel: int,
        is_open: bool,
    ) -> bool:
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "kill-switch-btn":
            return True
        if trigger_id == "kill-switch-confirm":
            if engine is not None:
                engine.stop()
            return False
        if trigger_id == "kill-switch-cancel":
            return False
        return is_open

    # ------------------------------------------------------------------
    # Performance tab
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("perf-equity-chart", "figure"),
            Output("perf-drawdown-chart", "figure"),
            Output("perf-summary-table", "children"),
            Output("monthly-heatmap", "figure"),
            Output("pnl-histogram", "figure"),
            Output("perf-total-pnl", "children"),
            Output("perf-sharpe", "children"),
            Output("perf-max-dd", "children"),
            Output("perf-win-rate", "children"),
        ],
        Input("perf-interval", "n_intervals"),
    )
    def update_performance_tab(n: int):  # noqa: ANN001
        db_path = cfg.database.db_path
        _colors = ["#f5a623", "#4488ff", "#00e676", "#ff5252", "#aa44ff"]

        # Equity curve — all time, one line per strategy
        try:
            eq_df = queries.get_equity_curve(db_path)
        except Exception:
            eq_df = pd.DataFrame()

        if not eq_df.empty:
            eq_df["exit_time"] = pd.to_datetime(eq_df["exit_time"])
            eq_fig = go.Figure()
            for i, (sid, grp) in enumerate(eq_df.groupby("strategy_id")):
                grp = grp.sort_values("exit_time")
                eq_fig.add_trace(go.Scatter(
                    x=grp["exit_time"],
                    y=grp["cumulative_pnl"],
                    mode="lines",
                    name=str(sid),
                    line=dict(color=_colors[i % len(_colors)], width=2),
                ))
            eq_fig.update_layout(_dark_layout(title="Equity Curve (All Time)"))
        else:
            eq_fig = _empty_figure()

        # Drawdown chart — computed from aggregate equity
        if not eq_df.empty:
            all_eq = eq_df.sort_values("exit_time").reset_index(drop=True)
            cum = all_eq["cumulative_pnl"].values
            peak = pd.Series(cum).cummax()
            drawdown = pd.Series(cum) - peak
            dd_fig = go.Figure(go.Scatter(
                x=all_eq["exit_time"].values,
                y=drawdown.values,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255,82,82,0.25)",
                line=dict(color="#ff5252", width=1),
                name="Drawdown",
            ))
            dd_fig.update_layout(_dark_layout(title="Drawdown"))
        else:
            dd_fig = _empty_figure()

        # Performance summary table
        try:
            perf_df = queries.get_performance_summary(db_path)
        except Exception:
            perf_df = pd.DataFrame()

        _perf_headers = ["Strategy", "Total PnL", "Trades", "Win Rate", "Sharpe", "Max DD"]
        _perf_thead = html.Thead(html.Tr([html.Th(h) for h in _perf_headers]))

        if perf_df.empty:
            summary_tbl: dbc.Table = dbc.Table([
                _perf_thead,
                html.Tbody([_empty_table_row("No trade data available", len(_perf_headers))]),
            ], bordered=True, dark=True, hover=True, size="sm")
            total_pnl_txt: object = "$0.00"
            sharpe_txt: object = "0.00"
            maxdd_txt: object = "$0.00"
            winrate_txt: object = "0.0%"
        else:
            rows = []
            for _, row in perf_df.iterrows():
                pnl_v = float(row.get("total_pnl", 0))
                rows.append(html.Tr([
                    html.Td(row.get("strategy_id", "")),
                    html.Td(f"${pnl_v:+,.2f}",
                            style={"color": _pnl_color(pnl_v)}),
                    html.Td(row.get("trade_count", 0)),
                    html.Td(f"{float(row.get('win_rate', 0)) * 100:.1f}%"),
                    html.Td(f"{float(row.get('sharpe', 0)):.2f}"),
                    html.Td(f"${float(row.get('max_drawdown', 0)):,.2f}"),
                ]))
            summary_tbl = dbc.Table([
                _perf_thead, html.Tbody(rows),
            ], bordered=True, dark=True, hover=True, size="sm")

            total_pnl = float(perf_df["total_pnl"].sum())
            avg_sharpe = float(perf_df["sharpe"].mean())
            min_dd = float(perf_df["max_drawdown"].min())
            avg_wr = float(perf_df["win_rate"].mean())

            total_pnl_txt = html.Span(
                f"${total_pnl:+,.2f}", style={"color": _pnl_color(total_pnl)}
            )
            sharpe_txt = f"{avg_sharpe:.2f}"
            maxdd_txt = f"${min_dd:,.2f}"
            winrate_txt = f"{avg_wr * 100:.1f}%"

        # Monthly PnL heatmap
        try:
            daily_df = queries.get_daily_pnl(db_path)
        except Exception:
            daily_df = pd.DataFrame()

        if not daily_df.empty:
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            agg = daily_df.groupby("date")["pnl"].sum().reset_index()
            agg["month"] = agg["date"].dt.strftime("%Y-%m")
            agg["day"] = agg["date"].dt.day
            pivot = agg.pivot_table(
                index="month", columns="day", values="pnl", aggfunc="sum"
            )
            heat_fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                colorbar=dict(title="PnL ($)"),
                hoverongaps=False,
            ))
            heat_fig.update_layout(_dark_layout(
                title="Monthly PnL Heatmap",
                xaxis=dict(title="Day of Month", gridcolor="#1e1e2a"),
                yaxis=dict(title="Month", gridcolor="#1e1e2a"),
            ))
        else:
            heat_fig = _empty_figure()

        # Win/loss histogram
        try:
            rt_df = queries.get_round_trips(db_path)
        except Exception:
            rt_df = pd.DataFrame()

        if not rt_df.empty:
            wins = rt_df.loc[rt_df["pnl"] > 0, "pnl"]
            losses = rt_df.loc[rt_df["pnl"] <= 0, "pnl"]
            hist_fig = go.Figure()
            if not wins.empty:
                hist_fig.add_trace(go.Histogram(
                    x=wins, name="Wins",
                    marker_color="#00e676", opacity=0.7,
                ))
            if not losses.empty:
                hist_fig.add_trace(go.Histogram(
                    x=losses, name="Losses",
                    marker_color="#ff5252", opacity=0.7,
                ))
            hist_fig.update_layout(_dark_layout(
                title="Win / Loss Distribution",
                barmode="overlay",
                xaxis=dict(title="PnL per Trade ($)", gridcolor="#1e1e2a"),
                yaxis=dict(title="Frequency", gridcolor="#1e1e2a"),
            ))
        else:
            hist_fig = _empty_figure()

        return (
            eq_fig, dd_fig, summary_tbl, heat_fig, hist_fig,
            total_pnl_txt, sharpe_txt, maxdd_txt, winrate_txt,
        )

    # ------------------------------------------------------------------
    # Health tab (shares perf-interval — 60 s is fine)
    # ------------------------------------------------------------------

    @app.callback(
        [
            Output("system-events-table", "children"),
            Output("signal-freq-chart", "figure"),
            Output("slippage-chart", "figure"),
            Output("macro-regime-panel", "children"),
            Output("upcoming-events-table", "children"),
        ],
        Input("perf-interval", "n_intervals"),
    )
    def update_health_tab(n: int):  # noqa: ANN001
        db_path = cfg.database.db_path

        # System events
        try:
            events_df = queries.get_system_events(db_path, limit=20)
        except Exception:
            events_df = pd.DataFrame()

        evt_headers = ["Time", "Type", "Message"]
        evt_thead = html.Thead(html.Tr([html.Th(h) for h in evt_headers]))
        if events_df.empty:
            events_tbl: dbc.Table = dbc.Table([
                evt_thead,
                html.Tbody([_empty_table_row("No system events", len(evt_headers))]),
            ], bordered=True, dark=True, size="sm")
        else:
            evt_rows = []
            for _, row in events_df.iterrows():
                kind = str(row.get("kind", row.get("type", "")))
                bg = (
                    "rgba(255,82,82,0.15)" if kind == "ERROR"
                    else ("rgba(245,166,35,0.15)" if kind == "WARNING" else "")
                )
                evt_rows.append(html.Tr([
                    html.Td(str(row.get("timestamp", ""))),
                    html.Td(kind),
                    html.Td(str(row.get("message", ""))),
                ], style={"backgroundColor": bg} if bg else {}))
            events_tbl = dbc.Table([
                evt_thead, html.Tbody(evt_rows),
            ], bordered=True, dark=True, size="sm", responsive=True)

        # Signal frequency — fills grouped by date + strategy
        try:
            all_fills = queries.get_fills(db_path)
        except Exception:
            all_fills = pd.DataFrame()

        if not all_fills.empty:
            all_fills["timestamp"] = pd.to_datetime(all_fills["timestamp"])
            all_fills["date"] = all_fills["timestamp"].dt.date
            freq = (
                all_fills
                .groupby(["date", "strategy_id"])
                .size()
                .reset_index(name="count")
            )
            _fc = ["#f5a623", "#4488ff", "#00e676", "#ff5252"]
            freq_fig = go.Figure()
            for i, (sid, grp) in enumerate(freq.groupby("strategy_id")):
                freq_fig.add_trace(go.Bar(
                    x=grp["date"],
                    y=grp["count"],
                    name=str(sid),
                    marker_color=_fc[i % len(_fc)],
                ))
            freq_fig.update_layout(_dark_layout(
                title="Fill Frequency by Day",
                barmode="stack",
                xaxis=dict(title="Date", gridcolor="#1e1e2a"),
                yaxis=dict(title="Fills", gridcolor="#1e1e2a"),
            ))
        else:
            freq_fig = _empty_figure()

        # Commission tracker
        try:
            rt_df = queries.get_round_trips(db_path)
        except Exception:
            rt_df = pd.DataFrame()

        if not rt_df.empty:
            avg_comm = (
                rt_df
                .groupby("strategy_id")["commission"]
                .mean()
                .reset_index()
            )
            slip_fig = go.Figure(go.Bar(
                x=avg_comm["strategy_id"],
                y=avg_comm["commission"],
                marker_color="#f5a623",
                name="Avg Commission",
            ))
            slip_fig.update_layout(_dark_layout(
                title="Avg Commission per Trade",
                xaxis=dict(title="Strategy"),
                yaxis=dict(title="Commission ($)"),
            ))
        else:
            slip_fig = _empty_figure()

        return (
            events_tbl,
            freq_fig,
            slip_fig,
            _build_regime_panel(),
            _build_upcoming_events_table(),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_app(eng, cfg: AppConfig) -> dash.Dash:
    """
    Construct and return the configured Dash application.

    Sets module-level ``engine`` and ``config`` references used by callbacks.
    Call ``app.run(...)`` or use :func:`run_dashboard` to serve.
    """
    global engine, config
    engine = eng
    config = cfg

    # Collect all registered strategy IDs at startup time so callback
    # inputs are static (required by Dash).
    strategy_ids: list[str] = []
    if eng is not None and hasattr(eng, "_strategies"):
        strategy_ids = [s.strategy_id for s in eng._strategies]

    # Determine paper vs live mode label for header subtitle
    _is_paper = True
    if cfg is not None:
        _is_paper = getattr(cfg, "paper_trading", True)
    _mode_label = "Paper Trading" if _is_paper else "Live Trading"
    _mode_color = "#00e676" if _is_paper else "#ff5252"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.SLATE],
        suppress_callback_exceptions=True,
        title="AlgoTrade",
    )

    app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #0d0d0d !important;
                color: #e8e8f0 !important;
            }
            .dash-graph, .js-plotly-plot, .plotly {
                background: transparent !important;
            }

            /* Tab bar */
            .nav-tabs {
                background-color: #0d0d0d;
                border-bottom: 1px solid #2a2a3a;
            }
            .nav-tabs .nav-link {
                color: #6b7280 !important;
                border: none !important;
                border-bottom: 2px solid transparent !important;
                background: transparent !important;
                padding: 10px 18px;
            }
            .nav-tabs .nav-link.active {
                color: #f5a623 !important;
                border-bottom: 2px solid #f5a623 !important;
                background: transparent !important;
            }
            .nav-tabs .nav-link:hover {
                color: #e8e8f0 !important;
                border-bottom: 2px solid #2a2a3a !important;
            }

            /* Navbar */
            .navbar, nav.navbar {
                background-color: #0d0d0d !important;
                border-bottom: 1px solid #2a2a3a !important;
            }

            /* Emergency stop hover */
            .kill-switch-btn:hover {
                background: #ff5252 !important;
                color: #fff !important;
            }

            /* Pulse animation for connected dot */
            @keyframes pulse {
                0%   { opacity: 1; }
                50%  { opacity: 0.35; }
                100% { opacity: 1; }
            }
            .connection-dot-connected {
                animation: pulse 2s ease-in-out infinite;
            }

            /* Tables */
            .table {
                color: #e8e8f0 !important;
            }
            .table thead th {
                background-color: #13131d !important;
                color: #6b7280 !important;
                font-size: 11px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.06em !important;
                border-bottom: 1px solid #2a2a3a !important;
                border-top: none !important;
            }
            .table td, .table th {
                border-color: #2a2a3a !important;
            }
            .table-dark {
                background-color: transparent !important;
            }
            .table-dark td, .table-dark th, .table-dark thead th {
                border-color: #2a2a3a !important;
            }
            .table tbody tr:nth-child(even) td {
                background-color: #1e1e2a !important;
            }
            .table-hover tbody tr:hover td {
                background-color: rgba(245,166,35,0.06) !important;
            }

            /* Modal */
            .modal-content {
                background-color: #1a1a24 !important;
                border: 1px solid #2a2a3a !important;
                color: #e8e8f0 !important;
            }
            .modal-header {
                border-bottom: 1px solid #2a2a3a !important;
            }
            .modal-footer {
                border-top: 1px solid #2a2a3a !important;
            }
            .modal-title {
                color: #e8e8f0 !important;
            }

            /* Cards global fallback */
            .card {
                background-color: #1a1a24 !important;
                border: 1px solid #2a2a3a !important;
            }

            /* Alerts */
            .alert-secondary {
                background-color: #1a1a24 !important;
                border-color: #2a2a3a !important;
                color: #6b7280 !important;
            }

            /* Container background */
            .container-fluid, .container {
                background-color: transparent !important;
            }

            /* Badge pill overrides (for callback-driven color updates) */
            .badge.bg-success {
                background-color: rgba(0,230,118,0.2) !important;
                color: #00e676 !important;
                border: 1px solid #00e676 !important;
                border-radius: 99px !important;
            }
            .badge.bg-danger {
                background-color: rgba(255,82,82,0.2) !important;
                color: #ff5252 !important;
                border: 1px solid #ff5252 !important;
                border-radius: 99px !important;
            }

            /* Button color overrides for callback-driven updates */
            .btn-warning {
                background: #f5a623 !important;
                color: #000 !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            .btn-success:not(.kill-switch-btn) {
                background: #f5a623 !important;
                color: #000 !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            .btn-secondary {
                background: #2a2a3a !important;
                color: #e8e8f0 !important;
                border: 1px solid #3a3a4a !important;
                border-radius: 8px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

    app.layout = html.Div([
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Span("AlgoTrade Dashboard", style={
                            "fontSize": "20px", "fontWeight": "500",
                            "color": "#e8e8f0", "letterSpacing": "0.02em",
                        }),
                        html.Span(
                            f"  {_mode_label}",
                            style={"fontSize": "12px", "color": _mode_color,
                                   "marginLeft": "12px", "fontWeight": "500"},
                        ),
                    ]),
                ], align="center", className="py-3"),
            ], fluid=True),
        ], style={
            "backgroundColor": "#0d0d0d",
            "borderBottom": "1px solid #2a2a3a",
            "marginBottom": "0",
        }),
        dbc.Container([
            dbc.Tabs(
                [_live_tab(strategy_ids), _performance_tab(), _health_tab()],
                id="main-tabs",
                active_tab="tab-live",
            ),
        ], fluid=True),
    ], style={"backgroundColor": "#0d0d0d", "minHeight": "100vh"})

    _register_callbacks(app, cfg, strategy_ids)
    return app


def run_dashboard(eng, cfg: AppConfig) -> None:
    """Create and serve the Dash dashboard (blocking call)."""
    app = create_app(eng, cfg)
    app.run(
        host=cfg.dashboard.host,
        port=cfg.dashboard.port,
        debug=False,
    )


# ---------------------------------------------------------------------------
# Public API reference card
# ---------------------------------------------------------------------------
#
# Module-level state
# ------------------
# engine : TradingEngine | None
#   Set by create_app(); read by all callbacks via closure.
# config : AppConfig | None
#   Set by create_app(); read by all callbacks via closure.
#
# create_app(eng: TradingEngine, cfg: AppConfig) -> dash.Dash
#   Build the three-tab Dash application.
#   Sets module-level engine + config, wires layout and callbacks.
#   Returns the Dash instance; caller must call app.run() to serve.
#
# run_dashboard(eng: TradingEngine, cfg: AppConfig) -> None
#   Convenience wrapper: create_app() then app.run() with config host/port.
#   Intended to run in a daemon thread alongside TradingEngine.run().
#
# Tabs
# ----
# tab-live        id="tab-live"        — live-interval (5 s)
# tab-performance id="tab-performance" — perf-interval (60 s)
# tab-health      id="tab-health"      — perf-interval (60 s)
#
# Key callback outputs
# --------------------
# live-equity-chart       go.Figure  today's cumulative PnL line
# status-daily-pnl        html       coloured dollar amount
# status-drawdown         html       coloured percentage
# status-connection       html       coloured dot + label
# status-halted           html       Active / HALTED text
# resume-btn style        dict       shown only when is_halted
# positions-table         dbc.Table  open positions from shared_state
# fills-table             dbc.Table  today's fills from DB
# strategy-controls       list       one card per strategy with toggle
# perf-equity-chart       go.Figure  all-time equity per strategy
# perf-drawdown-chart     go.Figure  drawdown filled area
# perf-summary-table      dbc.Table  one row per strategy
# monthly-heatmap         go.Figure  RdYlGn heatmap
# pnl-histogram           go.Figure  wins (green) + losses (red)
# system-events-table     dbc.Table  ERROR=red, WARNING=amber rows
# signal-freq-chart       go.Figure  stacked bars by day/strategy
# slippage-chart          go.Figure  avg commission per strategy
# macro-regime-panel      html.Div   regime cards (read-only)
# upcoming-events-table   dbc.Table  impact-coloured rows
