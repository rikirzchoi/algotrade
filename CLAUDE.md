# Algotrade — multi-strategy algorithmic trading system

## What this project is
A Python-based algorithmic trading system connecting to Interactive
Brokers via ibapi. It runs multiple strategies simultaneously, manages
risk automatically, logs everything to SQLite, and exposes a Dash
dashboard for monitoring.

## Tech stack
- Python 3.11+
- ibapi (Interactive Brokers API)
- Dash + Plotly (dashboard)
- SQLite via sqlite3 (database)
- pandas, numpy, ta (data + indicators)
- dash-bootstrap-components (UI)

## Project structure
The project is in early development. Currently only `test.py` exists — a
single-file prototype demonstrating IB connection, historical data loading,
SMA calculation, and bracket order placement. The target structure is:

```
algotrade/
├── broker.py              # ONLY file that imports ibapi
├── config.py              # All configuration and constants
├── main.py                # Entry point; wires engine + strategies
├── core/
│   ├── engine.py          # Event loop; dispatches events to strategies
│   ├── events.py          # All event dataclasses (BarEvent, SignalEvent, …)
│   └── risk.py            # Position sizing and risk checks
├── strategies/
│   └── sma_crossover.py   # Example strategy (ported from test.py)
├── database/
│   └── writer.py          # ONLY file that writes to SQLite
├── dashboard/
│   └── app.py             # Dash app for monitoring
└── test.py                # Original prototype (reference only)
```

## Key design rules — ALWAYS follow these
1. broker.py is the ONLY file that imports or touches ibapi
2. Every event flows through core/events.py dataclasses —
   no raw dicts passed between modules
3. All config lives in config.py — no magic numbers elsewhere
4. Database writes happen in database/writer.py only
5. Strategies never place orders — they return SignalEvents,
   the engine places orders
6. All times are America/New_York timezone
7. Never raise exceptions in on_bar() — catch and log instead

## Coding style
- Type annotations on all function signatures
- Docstrings on all classes and public methods
- No print() in production code — use Python logging module
- threading.Lock wherever shared state is accessed across threads

## Current build status
- [x] test.py — single-file prototype (reference only, not production)
- [ ] broker.py
- [ ] config.py
- [ ] core/events.py
- [ ] core/engine.py
- [ ] core/risk.py
- [ ] strategies/sma_crossover.py
- [ ] database/writer.py
- [ ] dashboard/app.py
