# Algotrade

A Python systematic-trading system built from scratch on Interactive Brokers —
and, more importantly, a disciplined **research process** for telling a real
trading edge from luck.

## The honest story

It started as a multi-strategy intraday bot. Then it was used the way a quant
*should* use one: to rigorously test whether its strategies actually make money.
The verdicts (all logged in [`DECISIONS.md`](DECISIONS.md)):

| Tested | Verdict |
|---|---|
| 4 technical strategies (ORB, Bollinger, momentum, EMA crossover) | no edge |
| **Diversified trend-following** (14 markets) | **validated** — significant + robust |
| Defense contract-award catalysts | no edge / Gate-2 artifact |
| Biotech FDA catalysts | untestable on free data (survivorship wall) |

**One validated edge out of the lot** — diversified trend-following, a modest
(Sharpe ~0.5) diversifier. Deployed as a **60% equity / 40% managed-futures
(DBMF)** core. Everything else was correctly rejected *before* risking money —
including a tempting pre-committed signal that threw t > 3 in-sample and was caught
as a regime artifact at out-of-sample validation. That discipline is the point.

## What's in here

- **Risk + execution:** `core/` (event-driven engine, the single ibapi broker),
  `risk/` (kill switches, fixed-fractional sizing).
- **Research/validation:** `backtesting/` — honest backtester (conservative costs,
  no look-ahead), portfolio simulator of the live stack, statistical significance
  (Probabilistic Sharpe Ratio), factor attribution; `research/` — prospective
  biotech catalyst collector + defense studies.
- **Doctrine:** [`TRADING_RULES.md`](TRADING_RULES.md) (operating rules),
  [`DECISIONS.md`](DECISIONS.md) (decision log), [`ROADMAP.md`](ROADMAP.md).
- **Tests:** 36 tests in `tests/`.

## Running things

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Tests
.venv/bin/python tests/test_risk_accounting.py        # (and the other test_*.py)

# Research
.venv/bin/python -m backtesting.trend_research        # the validated edge
.venv/bin/python -m backtesting.trend_robustness      # its robustness checks
.venv/bin/python -m research.catalyst_logger          # biotech collector (also weekly via cron)
```

## Status
Research-led. The systematic core (trend + equity) is validated and ready to
deploy via ETFs; the live `main.py` engine is retained for reference (its wired
strategies were rejected — see the note in that file). Further edge research is
gated on either patience (the prospective collector) or paid survivorship-clean
data.
