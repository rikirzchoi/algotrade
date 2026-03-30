---
name: ORB strategy — symbol selection and bar size decision
description: Why intraday_symbols was narrowed to AMZN only and ORB switched to 5-min bars
type: project
---

ORB strategy is running on 5-min bars (switched from 1-min) and AMZN only (switched from AAPL, MSFT, NVDA, AMZN).

**Why:** Back-test showed positive expectancy only on AMZN with 5-min bars. MSFT and NVDA showed negative expectancy; AAPL was breakeven. 5-min bars also have better data availability and smoother signals.

**How to apply:** Do not re-add MSFT, NVDA, or AAPL to intraday_symbols without paper trading data. Re-evaluate after ~4 weeks of paper trading (around 2026-04-25). If suggesting ORB config changes, default to 5-min bars.
