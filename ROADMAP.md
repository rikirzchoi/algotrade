# Algotrade — improvement roadmap

A living record of the system's quant-readiness work: what's done, what's
pending, and the known gaps. Born out of a senior-quant-style code review
(June 2026). The goal is to move from "a bot that trades" to "a research
process that can tell edge from luck, with risk controls that demonstrably
work."

**Operating discipline lives in [`TRADING_RULES.md`](TRADING_RULES.md)** — the
rulebook that governs go-live, sizing, change control, and the kill switch. Read
it before changing anything that trades.

Status legend: ✅ done · 🔶 in progress · ⬜ not started

---

## Phase 1 — make it honest (DONE)

The system *looked* institutional but several core guarantees didn't actually
hold. These are fixed and covered by tests in `tests/`.

- ✅ **Account-level kill switches now fire.** `RiskManager.record_close()` was
  never called, so `daily_pnl`/equity never updated and the daily-loss and
  drawdown halts could never trip. The engine now maintains an authoritative
  position ledger (`core/engine.py: _OpenPosition`, `_account_for_fill`) that
  pairs each exit fill with its entry and books realised PnL into the risk
  manager. An auto-halt alerts via Telegram + audit `SystemEvent` but does NOT
  force-liquidate (resting bracket stops keep protecting open positions).
- ✅ **FLAT / end-of-day flatten actually closes positions.** Previously a FLAT
  signal fell through `_handle_signal` and did nothing. Added
  `Broker.flatten_symbol()` (cancels the resting bracket legs, then market-closes)
  and wired FLAT handling in the engine.
- ✅ **Bars table de-duplicated + protected.** Every restart re-ingested history
  with plain `INSERT`, leaving the DB ~87% duplicate rows. Added an idempotent
  dedupe migration + `UNIQUE(symbol, timestamp, bar_size)` index +
  `INSERT OR IGNORE` (`database/writer.py`).
- ✅ **Backtester is no longer optimistically biased.**
  - Stop is assumed hit first when a bar spans both stop and target.
  - Market-style exits (stop/flatten/end-of-data) take adverse slippage;
    limit targets fill at the limit.
  - Sizing now uses the **same** `RiskManager.size_position` path as live
    (fixed-fractional core + macro overlay). It previously sized ~33% larger
    by skipping the 0.75 neutral-regime multiplier. `fixed_fractional` is now
    the single source of truth for the core math.
- ✅ **Bar-size fan-out fixed (live engine).** `_dispatch` routed bars by symbol
  only, so AMZN's 15-min (Bollinger) and 5-min (ORB) bars both reached *both*
  strategies, cross-contaminating their indicator histories. Now matches symbol
  AND bar size (`core/engine.py: _strategy_wants`); the portfolio backtest does
  the same. Tested in `tests/test_portfolio_backtest.py`.
- ✅ **Test suite: 30 tests** across `tests/test_risk_accounting.py` (11),
  `tests/test_backtest_fills.py` (6), `tests/test_metrics_significance.py` (4),
  `tests/test_broker_commissions.py` (4), `tests/test_portfolio_backtest.py` (5).
  Run any with `.venv/bin/python tests/<file>.py`.

---

## Phase 2 — make it provable (ON HOLD — revisit after paper-trading data)

- ✅ **Portfolio-level backtest.** `backtesting/portfolio.py`
  (`PortfolioBacktestEngine`) replays the real live stack over a merged
  multi-symbol / multi-bar-size stream: all strategies, the real `RiskManager`
  (duplicate-position + correlation/concentration filters, daily-loss and
  drawdown kill switches), unified `size_position`, and the shared `CostModel`.
  Round-trips are booked into the RiskManager so the **kill switch fires
  portfolio-wide** and blocks later entries, exactly as live. `PortfolioResult`
  feeds `BacktestMetrics` for portfolio-level significance + a per-strategy
  breakdown and a rejection-reason tally. Data via `add_series()` or
  `load_csv(path, symbol, bar_size)`. Tested in
  `tests/test_portfolio_backtest.py` (filters engage, kill switch fires,
  bar-size routing isolated). Shared pieces extracted: `backtesting/costs.py`
  (`CostModel`) and `backtesting/data_loader.py` (`parse_ohlcv_csv`), now used
  by both engines.
  - Still open: deflated Sharpe across the strategy suite (multiple-testing).
- ⬜ **Walk-forward / out-of-sample framework.** In-sample optimise,
  out-of-sample validate, report the degradation. No parameter should be
  trusted on in-sample results alone.
- ✅ **Statistical honesty in metrics** (`backtesting/metrics.py`):
  - Added Sharpe t-stat, 95% CI (Lo 2002 SE), and Probabilistic Sharpe Ratio
    (Bailey & López de Prado, skew/kurtosis-adjusted) + an `is_significant`
    gate (PSR ≥ 0.95 AND ≥30 trades AND ≥20 obs). `print_summary` now shows a
    significance block and flags "NO — treat as noise" for weak/small samples.
  - `annualised_return` no longer extrapolates short windows (< ~3 months
    returns the non-annualised total instead of e.g. "240%").
  - Tested in `tests/test_metrics_significance.py`.
  - Still open: deflated Sharpe across the *suite* of strategies (multiple-
    testing correction) — belongs with the portfolio backtest.
- ✅ **Realistic transaction costs.**
  - Live: real commissions now recorded — `execDetails` buffers each fill until
    `commissionReport` patches in the actual commission, then queues it (with a
    heartbeat safety-net flush and IBKR float-max-sentinel handling).
  - Backtest: each fill pays slippage **plus half the bid-ask spread**; short
    positions accrue a **borrow/financing charge** for the holding period
    (`BacktestTrade.borrow_cost`). Tested in `tests/test_broker_commissions.py`
    and `tests/test_backtest_fills.py`.
  - Still crude: market impact is a flat slippage %, not size-dependent.
- ✅ **Mark-to-market equity curve (backtest).** Both backtest engines now
  snapshot realised equity **plus unrealised P&L of open positions** each bar,
  so drawdown/Sharpe capture open-position dips, not just closes. (Live MTM
  drawdown would require streaming account NLV — separate, lower priority.)

---

## Phase 3 — make it quant-grade (PARTIAL)

- ✅ **Edge thesis per strategy — resolved (mostly by rejection).** Each of ORB,
  Bollinger-reversion, EMA(21/55) crossover, momentum breakout was put to the
  thesis test (inefficiency? counterparty? regime?) and **killed for having no
  answer** (see `DECISIONS.md`, 2026-06-23). The thesis-first discipline then drove
  the edge-hunt that produced exactly one validated edge (trend). The four
  strategies are retained only as reference for that verdict.
- ✅ **Performance attribution / factor exposure.** `backtesting/attribution.py`:
  `factor_attribution()` regresses returns on a benchmark → alpha (annualised),
  beta, R², correlation — i.e. how much return is just beta you could buy for
  free. Surfaced in `PortfolioResult.summary(benchmark_returns=...)`. Tested in
  `tests/test_attribution.py`.
- 🔶 **Portfolio construction.** Analysis tools done: per-strategy return
  correlation matrix and inverse-volatility (risk-parity proxy) weights
  (`backtesting/attribution.py`, shown in `summary()`). Still open: applying
  vol-targeting / correlation-aware allocation to the **live** sizing path
  (a deliberate live-path change, not just analysis).

---

## Phase 4 — edge research: CLOSED (consolidate, don't keep hunting)

The alpha hunt has reached its honest end state. The deliberate decision (2026-06-29)
is to **stop manufacturing edges and consolidate around what validated.** This is a
conclusion, not a pause — recorded so the project ends on a decision rather than a
loose thread.

**The honest yield of edge-hunting (full tally in `DECISIONS.md`):** one validated
edge — **diversified time-series trend** (modest, ~0.5 Sharpe, a *diversifier*) —
out of everything tested. Rejected: 4 retail technical patterns, defense post-award,
defense announcement-timing, cross-sectional sector momentum, rates carry. Real but
unusable: OFI (data + latency wall), volatility risk premium (real edge, but
ruin-shaped negative skew and *long* crash risk — the opposite of what this book
owns). Untestable on free data: biotech FDA catalysts (survivorship wall) — kept as
a slow prospective collector.

- ✅ **Premia-broadening sweep done.** Cross-sectional momentum
  (`backtesting/xsec_momentum.py`), carry (`backtesting/carry_research.py`), and the
  volatility risk premium (`backtesting/vrp_research.py`) were each tested on
  survivorship-clean free data with the same significance machinery. Three swings,
  zero additions. Verdict: the product is the **validated trend sleeve blended with
  equity diversification** (the 60/40 core, `DECISIONS.md` cont. 6), not a stack of
  retail signals.
- ⛔ **Not pursuing: a paid survivorship-free / point-in-time dataset.** It would
  unlock single-stock cross-sectional factors, but those are heavily arbitraged,
  would land near Sharpe ~0.5 *again*, and correlate to the trend sleeve already
  owned — real spend for a likely-duplicate edge. Revisit only as a *learning*
  exercise for a future quant role, not as an investment.
- 🔬 **Still simmering (zero-cost, patience is the moat):** the biotech FDA-catalyst
  collector (`research/`) keeps accumulating clean prospective data. The only thread
  left running, because it costs nothing and waiting *is* the strategy.

**The real deliverable is the process, not a signal.** The institutional-grade
result here is a validation discipline that kills its own ideas honestly — PSR /
deflated-Sharpe gates, survivorship awareness, refusing to p-hack a rescue, and
distinguishing a *real* premium (VRP) from a *right* one. That is `DECISIONS.md`,
and it is effectively complete.

---

## Known smaller gaps (not yet scheduled)

- ✅ **Real commissions recorded.** `Broker.commissionReport` now patches the
  reported commission into the buffered fill before it reaches the engine, so
  realised PnL includes commission.
- ✅ **Correlation filter fixed.** Was dead (required *all* of `("QQQ","SPY")`
  long, but SPY is never traded). Replaced with a count-based concentration
  rule: a new risk-on LONG is blocked once `max_risk_on_positions` (default 2)
  of `risk_on_symbols` (QQQ/MSFT/NVDA/AMZN) are already open; commodities are
  exempt. `config.py: RiskConfig`, `risk/manager.py: approve_signal`, tested in
  `tests/test_risk_accounting.py`.
- ⬜ **"1% risk per trade" isn't equalised.** With tight stops the
  `max_position_size` cap binds instead of the risk %, so per-trade risk varies
  widely across strategies. Revisit once portfolio risk is modelled.
- ⬜ **Sizing basis decision.** Live sizes off static `config.risk.capital`.
  Now that `RiskManager.current_equity` updates live, decide deliberately
  whether to compound off current equity instead.
- ✅ **pandas FutureWarning fixed** in `dashboard/server.py` — now fills only
  numeric columns instead of `df.fillna(0)` on the whole frame.

---

## How to run (reminder)

1. Start TWS, log in to the paper account (port 7497).
2. `./run.sh` — starts engine + dashboard, logs to `logs/`, auto-restarts on
   disconnect.
3. Dashboard at http://localhost:3000.
4. Tests: `.venv/bin/python tests/test_risk_accounting.py` and
   `.venv/bin/python tests/test_backtest_fills.py`.
