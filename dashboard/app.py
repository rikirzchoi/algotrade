"""
dashboard.app — sidebar-navigation Dash monitoring interface.

Page 1 — Overview
    Real-time equity curve, open positions, fills, strategy toggles,
    kill-switch in sidebar.

Page 2 — Performance
    Cumulative P&L, drawdown, per-strategy breakdown, monthly heatmap,
    win/loss histogram.

Page 3 — Health
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
    """Base chart layout dict merged into every figure."""
    base: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#888888"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(
            bgcolor="#1c1c1c",
            bordercolor="#2a2a3a",
            font=dict(color="#f0f0f0", size=13, family="Inter"),
        ),
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
    return "#00c853" if value >= 0 else "#ff1744"


# ---------------------------------------------------------------------------
# Layout style constants
# ---------------------------------------------------------------------------

_CARD_STYLE: dict = {
    "background": "var(--card)",
    "border": "var(--card-border)",
    "borderRadius": "16px",
    "padding": "20px 24px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.07)",
    "height": "100%",
}

_CHART_CONTAINER_STYLE: dict = {
    "background": "var(--card)",
    "border": "var(--card-border)",
    "borderRadius": "16px",
    "padding": "8px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.07)",
}

_BADGE_LONG = {
    "backgroundColor": "rgba(0,200,83,0.15)", "color": "#00c853",
    "border": "1px solid #00c853", "borderRadius": "99px",
    "padding": "2px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BADGE_SHORT = {
    "backgroundColor": "rgba(255,23,68,0.15)", "color": "#ff1744",
    "border": "1px solid #ff1744", "borderRadius": "99px",
    "padding": "2px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BADGE_ACTIVE_STYLE = {
    "backgroundColor": "rgba(0,200,83,0.2)", "color": "#00c853",
    "border": "1px solid #00c853", "borderRadius": "99px",
    "padding": "3px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BADGE_INACTIVE_STYLE = {
    "backgroundColor": "rgba(255,23,68,0.2)", "color": "#ff1744",
    "border": "1px solid #ff1744", "borderRadius": "99px",
    "padding": "3px 10px", "fontSize": "11px", "fontWeight": "600",
}
_BTN_TOGGLE_STYLE = {
    "background": "transparent",
    "border": "1px solid #6b7280",
    "color": "#6b7280",
    "borderRadius": "8px",
    "fontWeight": "600",
    "padding": "4px 14px",
    "fontSize": "12px",
}
_CARD_STRATEGY_STYLE = {
    "background": "var(--card)",
    "border": "var(--card-border)",
    "borderRadius": "12px",
}


def _stat_card(label: str, value_id: str, accent_color: str, default: str = "—",
               accent_class: str = "") -> dbc.Col:
    """Stat card with coloured icon accent, uppercase label, and large value."""
    return dbc.Col(html.Div([
        html.P(label, className="stat-card-label"),
        html.H3(id=value_id, children=default, className="stat-card-value mb-0"),
    ], className=f"stat-card {accent_class}"), width=3)


# ---------------------------------------------------------------------------
# Table builders (shared across callbacks)
# ---------------------------------------------------------------------------

def _build_positions_table(positions: dict) -> dbc.Table:
    headers = [html.Th("Symbol"), html.Th("Quantity"), html.Th("Side")]
    if not positions:
        return dbc.Table([
            html.Thead(html.Tr(headers)),
            html.Tbody([_empty_table_row("No open positions", 3)]),
        ], bordered=False, hover=True, size="sm")

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
    ], bordered=False, hover=True, size="sm")


def _build_fills_table(fills_df: pd.DataFrame) -> dbc.Table:
    col_headers = ["Time", "Strategy", "Symbol", "Side", "Qty", "Price"]
    thead = html.Thead(html.Tr([html.Th(h) for h in col_headers]))

    if fills_df.empty:
        return dbc.Table([
            thead,
            html.Tbody([_empty_table_row("No fills today", len(col_headers))]),
        ], bordered=False, hover=True, size="sm")

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
    ], bordered=False, hover=True, size="sm", responsive=True)


def _build_static_strategy_controls(strategy_ids: list[str]) -> list:
    """Build strategy control cards with stable component IDs (rendered once at startup)."""
    if not strategy_ids:
        return [html.P("No strategies registered.", className="text-muted")]
    cards = []
    for sid in strategy_ids:
        cards.append(dbc.Card(dbc.CardBody(dbc.Row([
            dbc.Col(html.Strong(sid, style={
                "color": "var(--text)", "fontWeight": "500", "fontSize": "14px",
            }), width=4),
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
        ], align="center")), className="mb-2 strategy-card", style=_CARD_STRATEGY_STYLE))
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
            dbc.Col(html.Strong(sid, style={
                "color": "var(--text)", "fontWeight": "500", "fontSize": "14px",
            }), width=4),
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
        ], align="center")), className="mb-2 strategy-card", style=_CARD_STRATEGY_STYLE))

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
        ], bordered=False, size="sm")

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
            "#ff1744" if impact == "high"
            else ("#ff9800" if impact == "medium" else "#00c853")
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
        ], bordered=False, size="sm")

    return dbc.Table([thead, html.Tbody(rows)], bordered=False, size="sm")


# ---------------------------------------------------------------------------
# Page layouts (replace dbc.Tab children with html.Div pages)
# ---------------------------------------------------------------------------

def _overview_page(strategy_ids: list[str] | None = None) -> html.Div:
    """Page 1 — Overview: stat cards, equity curve, fills, positions, strategy controls."""
    return html.Div(id="page-overview", className="page-content", children=[
        dcc.Interval(id="live-interval", interval=5000, n_intervals=0),

        # Row 1 — 4 Stat cards
        dbc.Row([
            _stat_card("Daily PnL", "status-daily-pnl", "#ff6b2b", "$0.00", "card-accent-orange"),
            _stat_card("Drawdown",  "status-drawdown",  "#ff1744", "0.00%", "card-accent-red"),
            _stat_card("Win Rate",  "perf-win-rate",    "#888888", "0.0%",  "card-accent-muted"),
            _stat_card("Total PnL", "perf-total-pnl",   "#00c853", "$0.00", "card-accent-green"),
        ], className="mb-4 mt-3", style={"--bs-gutter-x": "16px"}),

        # Row 2 — Live equity curve
        dbc.Row([
            dbc.Col(html.Div(
                dcc.Graph(id="live-equity-chart", figure=_empty_figure()),
                className="chart-card",
            ), width=12),
        ], className="mb-4"),

        # Row 3 — Fills + Positions
        dbc.Row([
            dbc.Col([
                html.H5("Today's Fills", className="mb-2 section-title"),
                html.Div(id="fills-table"),
            ], width=7),
            dbc.Col([
                html.H5("Open Positions", className="mb-2 section-title"),
                html.Div(id="positions-table"),
            ], width=5),
        ], className="mb-4"),

        # Row 4 — Strategy controls
        dbc.Row([
            dbc.Col([
                html.H5("Strategy Controls", className="mb-2 section-title"),
                html.Div(
                    _build_static_strategy_controls(strategy_ids or []),
                    id="strategy-controls",
                ),
                html.Div(id="strategy-toggle-output", style={"display": "none"}),
            ], width=12),
        ], className="mb-4"),
    ])


def _performance_page() -> html.Div:
    """Page 2 — Performance: charts, breakdown table, heatmap, histogram."""
    return html.Div(id="page-performance", className="page-content",
                    style={"display": "none"}, children=[
        dcc.Interval(id="perf-interval", interval=60_000, n_intervals=0),

        # Row 1 — 2 summary cards (win-rate + total-pnl live in overview page)
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(style={
                    "width": "36px", "height": "36px", "borderRadius": "8px",
                    "backgroundColor": "#f5a62326",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "marginBottom": "12px",
                }, children=html.Div(style={
                    "width": "14px", "height": "14px", "borderRadius": "3px",
                    "backgroundColor": "#f5a623",
                })),
                html.P("SHARPE RATIO", style={
                    "fontSize": "11px", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "var(--text-muted)",
                    "marginBottom": "6px", "fontWeight": "500",
                }),
                html.H3(id="perf-sharpe", children="0.00", className="mb-0", style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": "var(--text)", "lineHeight": "1.1",
                }),
            ], style=_CARD_STYLE), width=6),

            dbc.Col(html.Div([
                html.Div(style={
                    "width": "36px", "height": "36px", "borderRadius": "8px",
                    "backgroundColor": "#ff174426",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "marginBottom": "12px",
                }, children=html.Div(style={
                    "width": "14px", "height": "14px", "borderRadius": "3px",
                    "backgroundColor": "#ff1744",
                })),
                html.P("MAX DRAWDOWN", style={
                    "fontSize": "11px", "textTransform": "uppercase",
                    "letterSpacing": "0.08em", "color": "var(--text-muted)",
                    "marginBottom": "6px", "fontWeight": "500",
                }),
                html.H3(id="perf-max-dd", children="$0.00", className="mb-0", style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": "#ff1744", "lineHeight": "1.1",
                }),
            ], style=_CARD_STYLE), width=6),
        ], className="mb-4 mt-3", style={"--bs-gutter-x": "16px"}),

        # Row 2 — Equity + Drawdown charts
        dbc.Row([
            dbc.Col(html.Div(
                dcc.Graph(id="perf-equity-chart", figure=_empty_figure()),
                className="chart-card",
            ), width=6),
            dbc.Col(html.Div(
                dcc.Graph(id="perf-drawdown-chart", figure=_empty_figure()),
                className="chart-card",
            ), width=6),
        ], className="mb-4"),

        # Row 3 — Per-strategy table
        dbc.Row([
            dbc.Col([
                html.H5("Per-Strategy Breakdown", className="mb-2 section-title"),
                html.Div(id="perf-summary-table"),
            ], width=12),
        ], className="mb-4"),

        # Row 4 — Monthly heatmap
        dbc.Row([
            dbc.Col([
                html.H5("Monthly PnL Heatmap", className="mb-2 section-title"),
                html.Div(
                    dcc.Graph(id="monthly-heatmap", figure=_empty_figure()),
                    className="chart-card",
                ),
            ], width=12),
        ], className="mb-4"),

        # Row 5 — Win/loss histogram
        dbc.Row([
            dbc.Col([
                html.H5("Win / Loss Distribution", className="mb-2 section-title"),
                html.Div(
                    dcc.Graph(id="pnl-histogram", figure=_empty_figure()),
                    className="chart-card",
                ),
            ], width=12),
        ], className="mb-4"),
    ])


def _health_page() -> html.Div:
    """Page 3 — Health: system events, signal freq, commission, macro, upcoming."""
    return html.Div(id="page-health", className="page-content",
                    style={"display": "none"}, children=[
        # Row 1 — System events
        dbc.Row([
            dbc.Col([
                html.H5("System Events", className="mb-2 mt-3 section-title"),
                html.Div(id="system-events-table"),
            ], width=12),
        ], className="mb-4"),

        # Row 2 — Signal frequency + Commission tracker
        dbc.Row([
            dbc.Col([
                html.H5("Signal Frequency", className="mb-2 section-title"),
                html.Div(
                    dcc.Graph(id="signal-freq-chart", figure=_empty_figure()),
                    className="chart-card",
                ),
            ], width=6),
            dbc.Col([
                html.H5("Avg Commission per Trade", className="mb-2 section-title"),
                html.Div(
                    dcc.Graph(id="slippage-chart", figure=_empty_figure()),
                    className="chart-card",
                ),
            ], width=6),
        ], className="mb-4"),

        # Row 3 — Macro regime
        dbc.Row([
            dbc.Col([
                html.H5("Macro Regime", className="mb-2 section-title"),
                html.Div(id="macro-regime-panel"),
            ], width=12),
        ], className="mb-4"),

        # Row 4 — Upcoming events
        dbc.Row([
            dbc.Col([
                html.H5("Upcoming Events", className="mb-2 section-title"),
                html.Div(id="upcoming-events-table"),
            ], width=12),
        ], className="mb-4"),
    ])


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
    # Sidebar navigation — page switching
    # ------------------------------------------------------------------

    _NAV_ACTIVE_STYLE = {
        "background": "#2d2d3d",
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "10px 16px",
        "cursor": "pointer",
        "fontSize": "14px",
        "fontWeight": "600",
        "marginBottom": "4px",
        "userSelect": "none",
        "transition": "background 0.15s",
    }
    _NAV_INACTIVE_STYLE = {
        "background": "transparent",
        "color": "#9ca3af",
        "borderRadius": "8px",
        "padding": "10px 16px",
        "cursor": "pointer",
        "fontSize": "14px",
        "marginBottom": "4px",
        "userSelect": "none",
        "transition": "background 0.15s",
    }

    @app.callback(
        Output("current-page", "data"),
        [
            Input("nav-overview", "n_clicks"),
            Input("nav-performance", "n_clicks"),
            Input("nav-health", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def set_active_page(n1: int, n2: int, n3: int) -> str:
        ctx = callback_context
        if not ctx.triggered:
            return "overview"
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return {
            "nav-overview": "overview",
            "nav-performance": "performance",
            "nav-health": "health",
        }.get(btn_id, "overview")

    @app.callback(
        [
            Output("page-overview", "style"),
            Output("page-performance", "style"),
            Output("page-health", "style"),
            Output("nav-overview", "style"),
            Output("nav-performance", "style"),
            Output("nav-health", "style"),
            Output("page-header-title", "children"),
        ],
        Input("current-page", "data"),
    )
    def switch_page(page: str):  # noqa: ANN001
        show = {"display": "block"}
        hide = {"display": "none"}
        titles = {"overview": "Overview", "performance": "Performance", "health": "Health"}
        return (
            show if page == "overview" else hide,
            show if page == "performance" else hide,
            show if page == "health" else hide,
            _NAV_ACTIVE_STYLE if page == "overview" else _NAV_INACTIVE_STYLE,
            _NAV_ACTIVE_STYLE if page == "performance" else _NAV_INACTIVE_STYLE,
            _NAV_ACTIVE_STYLE if page == "health" else _NAV_INACTIVE_STYLE,
            titles.get(page, "Overview"),
        )

    # ------------------------------------------------------------------
    # Theme toggle
    # ------------------------------------------------------------------

    @app.callback(
        Output("theme-store", "data"),
        Input("theme-toggle", "value"),
        prevent_initial_call=True,
    )
    def update_theme(is_dark: bool) -> str:
        return "dark" if is_dark else "light"

    app.clientside_callback(
        """
        function(theme) {
            document.body.className = theme === 'dark' ? 'dark-mode' : 'light-mode';
            return '';
        }
        """,
        Output("theme-applier", "children"),
        Input("theme-store", "data"),
    )

    # ------------------------------------------------------------------
    # Live tab (Overview page)
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
            "#ff1744" if dd_val > 4
            else ("#ff9800" if dd_val > 2 else "var(--text)")
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
            style={"color": "#00c853" if connected else "#ff1744"},
        )

        # Kill-switch / halted
        is_halted: bool = bool(state.get("is_halted", False))
        if is_halted:
            halted_content = html.Span("HALTED",
                                       style={"color": "#ff1744", "fontWeight": "bold"})
            resume_style = {
                "display": "inline-block",
                "background": "#f5a623", "color": "#000",
                "border": "none", "borderRadius": "8px",
                "fontWeight": "600", "padding": "6px 16px",
            }
        else:
            halted_content = html.Span("Active", style={"color": "#00c853"})
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
                    colors = ["#ff6b2b", "#4488ff", "#00c853", "#ff1744", "#aa44ff"]
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
    # Performance page
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
        _colors = ["#ff6b2b", "#4488ff", "#00c853", "#ff1744", "#aa44ff"]

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
                fillcolor="rgba(255,23,68,0.25)",
                line=dict(color="#ff1744", width=1),
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
            ], bordered=False, hover=True, size="sm")
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
            ], bordered=False, hover=True, size="sm")

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
                    marker_color="#00c853", opacity=0.7,
                ))
            if not losses.empty:
                hist_fig.add_trace(go.Histogram(
                    x=losses, name="Losses",
                    marker_color="#ff1744", opacity=0.7,
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
    # Health page (shares perf-interval — 60 s is fine)
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
            ], bordered=False, size="sm")
        else:
            evt_rows = []
            for _, row in events_df.iterrows():
                kind = str(row.get("kind", row.get("type", "")))
                bg = (
                    "rgba(255,23,68,0.15)" if kind == "ERROR"
                    else ("rgba(255,152,0,0.15)" if kind == "WARNING" else "")
                )
                evt_rows.append(html.Tr([
                    html.Td(str(row.get("timestamp", ""))),
                    html.Td(kind),
                    html.Td(str(row.get("message", ""))),
                ], style={"backgroundColor": bg} if bg else {}))
            events_tbl = dbc.Table([
                evt_thead, html.Tbody(evt_rows),
            ], bordered=False, size="sm", responsive=True)

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
            _fc = ["#ff6b2b", "#4488ff", "#00c853", "#ff1744"]
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
                marker_color="#ff6b2b",
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

_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ============================================================
   ALGOTRADE DASHBOARD — Vaulto-inspired Theme
   ============================================================ */

/* ── Design Tokens (Dark Mode — default) ─────────────────── */
:root {
  --bg-main:          #0e0e0e;
  --bg-sidebar:       #141414;
  --bg-card:          #1c1c1c;
  --bg-card-hover:    #202020;
  --bg-input:         #1a1a1a;
  --bg-row-hover:     rgba(255,255,255,0.03);
  --bg-nav-hover:     rgba(255,255,255,0.05);
  --border-card:      rgba(255,255,255,0.05);
  --border-input:     rgba(255,255,255,0.08);
  --border-divider:   rgba(255,255,255,0.06);
  --text-primary:     #f0f0f0;
  --text-secondary:   #888888;
  --text-table-header:#666666;
  --text-muted:       #555555;
  --accent-orange:    #ff6b2b;
  --accent-green:     #00c853;
  --accent-red:       #ff1744;
  --accent-amber:     #ff9800;
  --chart-line:       #ff6b2b;
  --chart-grid:       rgba(255,255,255,0.04);
  --card-accent-pnl-daily:  #ff6b2b;
  --card-accent-drawdown:   #ff1744;
  --card-accent-winrate:    rgba(255,255,255,0.15);
  --card-accent-total-pnl:  #00c853;
  --sidebar-width:    220px;
  --sidebar-bg:       #141414;
  --font-family:      'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --radius-sm:        6px;
  --radius-md:        10px;
  --radius-lg:        12px;
  --radius-xl:        16px;
  --radius-pill:      999px;
  --shadow-card:      0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
  --shadow-card-hover:0 4px 12px rgba(0,0,0,0.5);
  --shadow-dropdown:  0 8px 24px rgba(0,0,0,0.6);
  --transition-fast:  0.15s ease;
  --transition-color: 0.2s ease;
  /* legacy aliases kept for any remaining var() refs in inline styles */
  --bg:               #0e0e0e;
  --card:             #1c1c1c;
  --card-border:      1px solid rgba(255,255,255,0.05);
  --text:             #f0f0f0;
  --text-muted-old:   #888888;
  --grid-color:       rgba(255,255,255,0.04);
  --table-alt:        rgba(255,255,255,0.02);
  --table-text:       #f0f0f0;
  --table-header-bg:  transparent;
  --table-header-txt: #666666;
  --modal-bg:         #1c1c1c;
  --modal-border:     rgba(255,255,255,0.05);
}

/* ── Light Mode Overrides ─────────────────────────────────── */
body.light-mode {
  --bg-main:          #f5f5f5;
  --bg-card:          #ffffff;
  --bg-card-hover:    #fafafa;
  --bg-input:         #f0f0f0;
  --bg-row-hover:     rgba(0,0,0,0.02);
  --bg-nav-hover:     rgba(0,0,0,0.04);
  --border-card:      rgba(0,0,0,0.08);
  --border-input:     rgba(0,0,0,0.1);
  --border-divider:   rgba(0,0,0,0.06);
  --text-primary:     #1a1a1a;
  --text-secondary:   #6b7280;
  --text-table-header:#9ca3af;
  --text-muted:       #9ca3af;
  --chart-grid:       rgba(0,0,0,0.04);
  --shadow-card:      0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-card-hover:0 4px 12px rgba(0,0,0,0.12);
  --shadow-dropdown:  0 8px 24px rgba(0,0,0,0.15);
  /* accents unchanged */
  --accent-orange:    #ff6b2b;
  --accent-green:     #00c853;
  --accent-red:       #ff1744;
  --accent-amber:     #ff9800;
  /* sidebar always dark */
  --sidebar-bg:       #141414;
  /* legacy aliases */
  --bg:               #f5f5f5;
  --card:             #ffffff;
  --card-border:      1px solid rgba(0,0,0,0.08);
  --text:             #1a1a1a;
  --grid-color:       rgba(0,0,0,0.04);
  --table-alt:        rgba(0,0,0,0.02);
  --table-text:       #1a1a1a;
  --table-header-bg:  transparent;
  --table-header-txt: #9ca3af;
  --modal-bg:         #ffffff;
  --modal-border:     rgba(0,0,0,0.08);
}

/* ── Base & Reset ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
  font-size: 16px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

body {
  font-family: var(--font-family);
  background-color: var(--bg-main);
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
  min-height: 100vh;
  transition: background-color var(--transition-color), color var(--transition-color);
}

/* ── Layout Shell ─────────────────────────────────────────── */
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background-color: var(--bg-main);
}

.main-content {
  flex: 1;
  margin-left: var(--sidebar-width);
  overflow-y: auto;
  overflow-x: hidden;
  padding: 32px;
  background-color: var(--bg-main);
  transition: background-color var(--transition-color);
}

.main-content::-webkit-scrollbar { width: 4px; }
.main-content::-webkit-scrollbar-track { background: transparent; }
.main-content::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.08);
  border-radius: var(--radius-pill);
}
body.light-mode .main-content::-webkit-scrollbar-thumb {
  background: rgba(0,0,0,0.12);
}

/* ── Sidebar ─────────────────────────────────────────────── */
.sidebar, #sidebar {
  position: fixed;
  top: 0; left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  background-color: #141414 !important;
  border-right: 1px solid rgba(255,255,255,0.05);
  display: flex;
  flex-direction: column;
  padding: 24px 0;
  z-index: 100;
  overflow-y: auto;
  overflow-x: hidden;
}
body.light-mode .sidebar,
body.light-mode #sidebar {
  background-color: #141414 !important;
  border-right: 1px solid rgba(255,255,255,0.05);
}
.sidebar::-webkit-scrollbar { width: 0; }

/* App name */
.sidebar-app-name {
  font-family: var(--font-family);
  font-size: 18px;
  font-weight: 700;
  color: #ff6b2b;
  letter-spacing: -0.01em;
}

/* Sidebar section label */
.sidebar-section-label {
  font-family: var(--font-family);
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: #444444 !important;
  padding: 0 16px;
  margin-top: 16px;
  margin-bottom: 4px;
}

/* Nav links */
.nav-link, .sidebar-nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  font-family: var(--font-family);
  font-size: 14px;
  font-weight: 500;
  color: #888888;
  border-left: 3px solid transparent;
  cursor: pointer;
  transition: background-color var(--transition-fast), color var(--transition-fast), border-left-color var(--transition-fast);
  user-select: none;
}
.nav-link:hover, .sidebar-nav-item:hover {
  background-color: rgba(255,255,255,0.05);
  color: #cccccc;
  text-decoration: none;
}
.nav-link.active, .sidebar-nav-item.active,
.nav-link.active-nav, .sidebar-nav-item.active-nav {
  color: #f0f0f0;
  border-left-color: #ff6b2b;
  background-color: rgba(255,107,43,0.06);
}

/* Nav hover override for ID-targeted links */
#nav-overview:hover, #nav-performance:hover, #nav-health:hover {
  background: rgba(255,255,255,0.05) !important;
  color: #cccccc !important;
}

/* Sidebar divider */
.sidebar-divider {
  height: 1px;
  background: rgba(255,255,255,0.05);
  margin: 16px;
}

/* Sidebar bottom */
.sidebar-bottom {
  margin-top: auto;
  padding: 16px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

/* Connection dot */
.connection-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background-color: #00c853;
  flex-shrink: 0;
  box-shadow: 0 0 6px rgba(0,200,83,0.6);
  animation: pulse-dot 2s ease-in-out infinite;
  display: inline-block;
}
.connection-dot.disconnected {
  background-color: #ff1744;
  box-shadow: 0 0 6px rgba(255,23,68,0.6);
  animation: none;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.6; transform: scale(0.9); }
}
/* legacy pulse for connection-dot-connected class */
.connection-dot-connected { animation: pulse-dot 2s ease-in-out infinite; }

/* Emergency stop button */
.btn-emergency-stop, .kill-switch-btn {
  width: 100% !important;
  padding: 9px 16px !important;
  background: transparent !important;
  border: 1px solid #ff1744 !important;
  border-radius: var(--radius-sm) !important;
  color: #ff1744 !important;
  font-family: var(--font-family) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em !important;
  cursor: pointer;
  transition: background-color var(--transition-fast), color var(--transition-fast), box-shadow var(--transition-fast) !important;
}
.btn-emergency-stop:hover, .kill-switch-btn:hover {
  background-color: #ff1744 !important;
  color: #ffffff !important;
  box-shadow: 0 0 12px rgba(255,23,68,0.35) !important;
}
.btn-emergency-stop:active, .kill-switch-btn:active {
  background-color: #cc1336 !important;
  transform: scale(0.98);
}

/* ── Page Header ──────────────────────────────────────────── */
.page-title {
  font-family: var(--font-family);
  font-size: 24px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  line-height: 1.2;
}

/* ── Stat Cards ───────────────────────────────────────────── */
.stat-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-card);
  height: 100%;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast), background-color var(--transition-color);
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background-color: var(--card-accent-color, rgba(255,255,255,0.15));
  border-radius: 3px 0 0 3px;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover); }

.card-accent-orange::before { background-color: #ff6b2b !important; }
.card-accent-red::before    { background-color: #ff1744 !important; }
.card-accent-green::before  { background-color: #00c853 !important; }
.card-accent-muted::before  { background-color: rgba(255,255,255,0.15) !important; }

.stat-card-label, .card-label {
  font-family: var(--font-family);
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-secondary);
  margin-bottom: 8px;
  line-height: 1;
}
.stat-card-value, .card-value {
  font-family: var(--font-family);
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
  line-height: 1;
  margin-bottom: 8px;
}
.stat-card-value.positive, .value-positive { color: #00c853; }
.stat-card-value.negative, .value-negative { color: #ff1744; }
.stat-card-value.warning                   { color: #ff9800; }

/* ── Generic Card ─────────────────────────────────────────── */
.card, .dbc-card {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-card) !important;
  color: var(--text-primary) !important;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast), background-color var(--transition-color);
}
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover) !important; }
.card-body  { padding: 20px 24px !important; }

/* ── Chart Card ───────────────────────────────────────────── */
.chart-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-lg);
  padding: 8px;
  box-shadow: var(--shadow-card);
  transition: transform var(--transition-fast), background-color var(--transition-color);
}
.chart-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover); }

.js-plotly-plot, .plotly, .dash-graph { background: transparent !important; }
.js-plotly-plot .plotly .main-svg     { background: transparent !important; }

/* ── Tables ───────────────────────────────────────────────── */
.table {
  color: var(--text-primary) !important;
  margin-bottom: 0;
  font-family: var(--font-family);
}
.table thead th {
  font-family: var(--font-family) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--text-table-header) !important;
  padding: 8px 16px !important;
  border-bottom: 1px solid var(--border-divider) !important;
  border-top: none !important;
  background: transparent !important;
}
.table td, .table th {
  border-color: var(--border-divider) !important;
  border-top: none !important;
  border-bottom: none !important;
}
.table tbody tr { height: 48px; transition: background-color var(--transition-fast); }
.table tbody tr:hover td { background-color: var(--bg-row-hover) !important; }
.table tbody td { font-size: 14px; color: var(--text-primary) !important; }
.table-dark, .table-dark td, .table-dark th, .table-dark thead th {
  background-color: transparent !important;
  color: var(--text-primary) !important;
  border-color: var(--border-divider) !important;
}

/* ── Strategy Card ────────────────────────────────────────── */
.strategy-card {
  background-color: var(--bg-card) !important;
  border: 1px solid var(--border-card) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-card) !important;
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}
.strategy-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover) !important; }

/* ── Badges ───────────────────────────────────────────────── */
.badge { font-family: var(--font-family) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.04em; padding: 3px 8px !important; border-radius: var(--radius-pill) !important; }
.badge.bg-success { background-color: rgba(0,200,83,0.12) !important; color: #00c853 !important; border: 1px solid #00c853 !important; }
.badge.bg-danger  { background-color: rgba(255,23,68,0.12) !important;  color: #ff1744 !important; border: 1px solid #ff1744 !important; }
.badge.bg-warning { background-color: rgba(255,152,0,0.12) !important;  color: #ff9800 !important; }
.badge.bg-primary { background-color: rgba(255,107,43,0.12) !important; color: #ff6b2b !important; }

/* ── Buttons ──────────────────────────────────────────────── */
.btn, button { font-family: var(--font-family) !important; cursor: pointer; border-radius: var(--radius-sm) !important; font-weight: 500 !important; transition: background-color var(--transition-fast), border-color var(--transition-fast), color var(--transition-fast), box-shadow var(--transition-fast), transform var(--transition-fast) !important; }
.btn-warning  { background: #ff6b2b !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.btn-success:not(.kill-switch-btn):not(.btn-emergency-stop) { background: #ff6b2b !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.btn-secondary { background: rgba(255,255,255,0.06) !important; color: var(--text-secondary) !important; border: 1px solid var(--border-card) !important; border-radius: 8px !important; }
.btn-secondary:hover { background: rgba(255,255,255,0.1) !important; color: var(--text-primary) !important; }

/* ── Form & Switch ────────────────────────────────────────── */
.form-check-input:checked { background-color: #ff6b2b !important; border-color: #ff6b2b !important; }
.form-switch .form-check-input { cursor: pointer; }

/* ── Modal ────────────────────────────────────────────────── */
.modal-content { background-color: var(--bg-card) !important; border: 1px solid var(--border-card) !important; border-radius: var(--radius-xl) !important; box-shadow: var(--shadow-dropdown) !important; color: var(--text-primary) !important; }
.modal-header  { border-bottom: 1px solid var(--border-divider) !important; padding: 20px !important; }
.modal-title   { font-size: 16px !important; font-weight: 600 !important; color: var(--text-primary) !important; }
.modal-body    { padding: 20px !important; }
.modal-footer  { border-top: 1px solid var(--border-divider) !important; padding: 12px 20px !important; }
.btn-close     { filter: invert(1) opacity(0.5) !important; }
.btn-close:hover { filter: invert(1) opacity(1) !important; }

/* ── Section title ────────────────────────────────────────── */
.section-title {
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-secondary);
  margin-bottom: 16px;
}

/* ── Alerts ───────────────────────────────────────────────── */
.alert-secondary { background-color: var(--bg-card) !important; border-color: var(--border-card) !important; color: var(--text-secondary) !important; }

/* ── Misc ─────────────────────────────────────────────────── */
.navbar, nav.navbar { display: none !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
.container-fluid, .container { background-color: transparent !important; }
.page-content { padding: 0 0 32px 0; }

/* ── Light mode card overrides ────────────────────────────── */
body.light-mode .card, body.light-mode .stat-card,
body.light-mode .chart-card, body.light-mode .strategy-card {
  background-color: #ffffff !important;
  border-color: rgba(0,0,0,0.08) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.05) !important;
}
body.light-mode .table, body.light-mode .table tbody td,
body.light-mode .table-dark, body.light-mode .table-dark td,
body.light-mode .table-dark th, body.light-mode .table-dark thead th {
  color: #1a1a1a !important;
}
body.light-mode .modal-content { background-color: #ffffff !important; border-color: rgba(0,0,0,0.08) !important; }
body.light-mode hr { border-color: rgba(0,0,0,0.06) !important; }
        </style>
    </head>
    <body class="dark-mode">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def create_app(eng, cfg: AppConfig) -> dash.Dash:
    """
    Construct and return the configured Dash application.

    Sets module-level ``engine`` and ``config`` references used by callbacks.
    Call ``app.run(...)`` or use :func:`run_dashboard` to serve.
    """
    global engine, config
    engine = eng
    config = cfg

    strategy_ids: list[str] = []
    if eng is not None and hasattr(eng, "_strategies"):
        strategy_ids = [s.strategy_id for s in eng._strategies]

    _is_paper = True
    if cfg is not None:
        _is_paper = getattr(cfg, "paper_trading", True)
    _mode_label = "Paper Trading" if _is_paper else "Live Trading"
    _mode_color = "#00c853" if _is_paper else "#ff1744"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="AlgoTrade",
    )

    app.index_string = _INDEX_STRING

    # ── Sidebar ────────────────────────────────────────────────────────
    sidebar = html.Div(
        id="sidebar",
        className="sidebar",
        children=[
            # Logo + mode badge
            html.Div([
                html.Div("AlgoTrade", className="sidebar-app-name", style={
                    "color": "#ff6b2b", "fontWeight": "700",
                }),
                html.Div(_mode_label, style={
                    "fontSize": "11px", "color": _mode_color,
                    "marginTop": "4px", "fontWeight": "600",
                    "letterSpacing": "0.04em", "textTransform": "uppercase",
                }),
            ], style={"padding": "28px 20px 20px"}),

            # Divider
            html.Hr(style={"margin": "0 16px 4px", "borderColor": "rgba(255,255,255,0.1)"}),

            # Nav links — initial style set here; callback updates on click
            html.Nav([
                html.Div("Overview", id="nav-overview", n_clicks=0, style={
                    "background": "#2d2d3d", "color": "#ffffff",
                    "borderRadius": "8px", "padding": "10px 16px",
                    "cursor": "pointer", "fontSize": "14px", "fontWeight": "600",
                    "marginBottom": "4px", "userSelect": "none",
                }),
                html.Div("Performance", id="nav-performance", n_clicks=0, style={
                    "background": "transparent", "color": "#9ca3af",
                    "borderRadius": "8px", "padding": "10px 16px",
                    "cursor": "pointer", "fontSize": "14px",
                    "marginBottom": "4px", "userSelect": "none",
                }),
                html.Div("Health", id="nav-health", n_clicks=0, style={
                    "background": "transparent", "color": "#9ca3af",
                    "borderRadius": "8px", "padding": "10px 16px",
                    "cursor": "pointer", "fontSize": "14px",
                    "marginBottom": "4px", "userSelect": "none",
                }),
            ], style={"padding": "12px 12px", "flex": "1"}),

            # Bottom: status + kill switch
            html.Div([
                html.Hr(style={"margin": "0 0 14px", "borderColor": "rgba(255,255,255,0.1)"}),

                # Connection status
                html.Div(id="status-connection",
                         children=html.Span("⬤ Unknown",
                                            style={"color": "rgba(255,255,255,0.4)"}),
                         style={"fontSize": "13px", "marginBottom": "8px"}),

                # Kill-switch status + resume
                html.Div([
                    html.Div(id="status-halted",
                             children=html.Span("Active", style={"color": "#00c853"}),
                             style={"fontSize": "13px", "marginBottom": "6px"}),
                    dbc.Button(
                        "Resume",
                        id="resume-btn",
                        size="sm",
                        n_clicks=0,
                        style={
                            "display": "none",
                            "background": "#f5a623", "color": "#000",
                            "border": "none", "borderRadius": "8px",
                            "fontWeight": "600", "padding": "4px 14px",
                            "fontSize": "12px", "marginBottom": "8px",
                        },
                    ),
                ]),

                html.Hr(style={"margin": "10px 0", "borderColor": "rgba(255,255,255,0.1)"}),

                # Emergency stop
                dbc.Button(
                    [html.Span("■ ", style={"color": "#ff1744", "marginRight": "6px"}),
                     "EMERGENCY STOP"],
                    id="kill-switch-btn",
                    className="kill-switch-btn btn-emergency-stop",
                    n_clicks=0,
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

            ], className="sidebar-bottom", style={"padding": "0 16px 24px", "marginTop": "auto"}),
        ],
        style={
            "width": "220px",
            "minHeight": "100vh",
            "background": "#1a1a2e",
            "display": "flex",
            "flexDirection": "column",
            "position": "fixed",
            "left": "0",
            "top": "0",
            "bottom": "0",
            "zIndex": "1000",
            "overflowY": "auto",
        },
    )

    # ── Main content ───────────────────────────────────────────────────
    main_content = html.Div(
        id="main-content",
        className="main-content",
        children=[
            # Top header bar
            html.Div([
                html.H5(id="page-header-title", children="Overview",
                        className="page-title", style={"margin": "0"}),
                # Theme toggle (right side)
                html.Div([
                    html.Span("Light", style={
                        "fontSize": "12px", "color": "var(--text-muted)",
                        "marginRight": "8px", "userSelect": "none",
                    }),
                    dbc.Switch(id="theme-toggle", value=False,
                               style={"display": "inline-block", "cursor": "pointer"}),
                    html.Span("Dark", style={
                        "fontSize": "12px", "color": "var(--text-muted)",
                        "marginLeft": "8px", "userSelect": "none",
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "20px 24px 8px",
                "borderBottom": "1px solid var(--card-border)",
                "marginBottom": "0",
                "background": "var(--bg)",
                "position": "sticky",
                "top": "0",
                "zIndex": "100",
            }),

            # Page containers (all always in DOM; display toggled by callback)
            html.Div([
                _overview_page(strategy_ids),
                _performance_page(),
                _health_page(),
            ], style={"padding": "0 24px"}),
        ],
        style={
            "marginLeft": "220px",
            "minHeight": "100vh",
            "background": "var(--bg-main)",
        },
    )

    app.layout = html.Div([
        dcc.Store(id="theme-store", data="dark"),
        dcc.Store(id="current-page", data="overview"),
        html.Div(id="theme-applier", style={"display": "none"}),
        sidebar,
        main_content,
    ], className="app-shell")

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
#   Build the sidebar-navigation Dash application.
#   Sets module-level engine + config, wires layout and callbacks.
#   Returns the Dash instance; caller must call app.run() to serve.
#
# run_dashboard(eng: TradingEngine, cfg: AppConfig) -> None
#   Convenience wrapper: create_app() then app.run() with config host/port.
