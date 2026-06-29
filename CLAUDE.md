# Algotrade — systematic trading + research system

## What this project is
A Python system connecting to Interactive Brokers (ibapi) that began as a
multi-strategy intraday bot and **evolved into a disciplined research process**
for finding and validating trading edges. It runs an event-driven engine, manages
risk automatically, logs to SQLite, and exposes a FastAPI + Next.js dashboard —
but its real value is the rigorous backtest/validation machinery and the honest
verdicts it has produced.

## Current reality (June 2026) — read the docs below for the real story
The honest outcome of rigorous testing:
- **One validated edge: diversified trend-following** (time-series momentum across
  ~14 ETF markets). Passed significance (PSR ~100%) + robustness (Gate 2). Modest
  (Sharpe ~0.5), a *diversifier*, not a return-maximiser.
- **Deployed plan: a 60% equity / 40% managed-futures (DBMF) core**, ETFs,
  quarterly rebalance. The bot is NOT needed for it.
- **Rejected (no edge, tested honestly):** the 4 original intraday/technical
  strategies (ORB, Bollinger, momentum-breakout, EMA crossover incl. GLD/USO),
  and a defense contract-award catalyst signal (Gate-2 artifact).
- **Background research:** a prospective biotech FDA-catalyst collector
  (`research/`) accumulating clean data (biotech can't be backtested on free data
  — survivorship wall). Other satellites (defense/chips/quantum) parked.

**Authoritative documents — consult before changing anything:**
- `TRADING_RULES.md` — the operating doctrine (risk limits, go-live gates, change
  control, kill-switch rules). Follow it.
- `DECISIONS.md` — the decision log: every verdict with date, evidence, and why.
- `ROADMAP.md` — what's built, what's pending.

## Tech stack
- Python 3.11+ (venv at `.venv`)
- ibapi (Interactive Brokers); yfinance + USAspending/openFDA for research data
- FastAPI + uvicorn backend; Next.js + TradingView dashboard (`dashboard-ui/`)
- SQLite; pandas, numpy, scipy, ta

## Project structure (actual)
```
core/         broker.py (ONLY ibapi file), engine.py, events.py, data.py
risk/         manager.py (risk gate + sizing), sizing.py
strategies/   the 4 rejected strategies (retained for reference; see DECISIONS.md)
backtesting/  engine.py, portfolio.py (live-stack sim), costs.py, metrics.py,
              attribution.py, data_loader.py, trend_research.py, trend_robustness.py
research/     catalyst_logger.py (biotech, runs weekly via cron), defense_study.py,
              defense_announcements.py
database/     writer.py (ONLY sqlite writer), queries.py
dashboard/    server.py (FastAPI); dashboard-ui/ (Next.js)
tests/        36 tests — run: .venv/bin/python tests/test_*.py
main.py       live engine entry (wires the now-rejected strategies; see note in file)
```

## Key design rules — ALWAYS follow these
1. core/broker.py is the ONLY file that imports or touches ibapi
2. Every event flows through core/events.py dataclasses — no raw dicts between modules
3. All config lives in config.py — no magic numbers elsewhere
4. Database writes happen in database/writer.py only
5. Strategies never place orders — they return SignalEvents; the engine places orders
6. All times are America/New_York timezone
7. Never raise exceptions in on_bar() — catch and log instead

## Research discipline — ALSO follow these (see TRADING_RULES.md)
8. Thesis first (a *why* + a counterparty), then backtest — never pattern-mine
9. A result needs out-of-sample + significance (PSR/deflated Sharpe), not a pretty curve
10. No result-driven parameter changes; pre-commit tests to avoid p-hacking
11. Survivorship-bias-free data only — free price data (yfinance) excludes failures

## Coding style
- Type annotations on all function signatures; docstrings on classes/public methods
- No print() in production code — use logging (research/CLI scripts may print)
- threading.Lock wherever shared state is accessed across threads
