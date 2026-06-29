# AlgoTrade — Trading Rules & Operating Doctrine

*Version 1.0 — drafted 2026-06-23. This is a living document, but it changes
**slowly and deliberately** (see §VI). It is version-controlled on purpose: every
change has a git history and a reason.*

---

## Preamble — why this document exists

This rulebook exists to **bind my future, emotional self with decisions made by
my present, rational self.**

When a position is underwater at 2am, when the account is down three weeks
running, when a strategy "feels" like it's about to turn — that version of me is
not to be trusted with decisions. *This* version of me, calm and thinking
clearly, writes the rules. The other version obeys them.

**The prime directive: survive first, profit second.** A blown-up account earns
0% forever. Every rule below serves survival before it serves return.

Three beliefs underpin everything here:

1. **Edge must be proven, never assumed.** A strategy makes money in this system
   only after it has demonstrated statistically significant edge on data it was
   not tuned on. Until then it is a hypothesis, not a strategy.
2. **The past does not determine the future.** Every backtest is a description of
   a world that already happened. Markets regime-shift, edges decay, and the map
   is not the territory. We trade small, we watch for divergence, and we never
   confuse a good backtest with a guarantee.
3. **Discipline beats intelligence.** The fastest way to lose is not a bad
   model — it's a good model overridden by a panicked human. The rules win.

---

## I. First principles

1. The strategies *propose*; the risk manager *disposes*; the rules govern the
   human. None of the three may be skipped.
2. Complexity must earn its keep with evidence. An unproven strategy is risk, not
   sophistication. When in doubt, **cut**, don't add.
3. We would rather trade **one** strategy we deeply understand than four we
   don't.
4. If I cannot explain *why* a strategy should have edge — what the inefficiency
   is and who is on the other side of the trade — I do not trade it with real
   money.

---

## II. Hard risk limits

These are enforced in code (`config.py` / `risk/manager.py`) **and** restated
here. Code and doctrine must always agree; if they ever diverge, **halt and
reconcile before trading.**

| Limit | Value | Enforced by |
|---|---|---|
| Risk per trade | ≤ 1% of capital (stop-defined) | `risk_per_trade_pct` |
| Max position size | ≤ 100 shares/symbol | `max_position_size` |
| Max daily loss | $500 (then auto-halt) | `max_daily_loss_usd` |
| Max drawdown from peak | 5% (then auto-halt) | `max_drawdown_pct` |
| Correlated risk-on positions | ≤ 2 open at once | `max_risk_on_positions` |

5. These numbers are **mine to set when calm**, and may only be changed via the
   change-control process (§VI) — never widened in the heat of a losing day to
   "give a trade room."
6. The percentages are the contract. If the dollar account grows, the dollar
   limits scale with it only by a deliberate, logged decision — not silently.
6a. ⚠️ **Calibration.** The dollar values above assume the configured `capital`.
    They are currently set for the **$100k paper** account and MUST be rescaled
    to the real account before going live. At **$1,000**: set `capital = 1000`,
    `max_daily_loss_usd ≈ 20–30` (not 500), keep `max_drawdown_pct = 0.05`
    (≈ $50), and lower `max_position_size` drastically — a 100-share cap on a
    $200 stock is a 20× over-leverage at $1,000. **Note:** the system caps
    position size by *share count*, not notional; before live at $1,000, prefer
    a notional cap or accept that one share can be 20–40% of the account.

6b. **Calibration table (set the dollar limits from capital).** Daily loss halt =
    the smaller of (3% of capital) or (half my pain threshold, ~$400–500).

    | Capital | Risk/trade (1%) | Daily-loss halt | Max drawdown (peak) |
    |---|---|---|---|
    | $1,000 (start) | ~$10 | ~$30 | ~$80 (8%) |
    | $5,000 | ~$50 | ~$150 | ~$400 |
    | $8,000 | ~$80 | ~$200 | ~$400–500 |

    Ordering that must always hold: **per-trade < daily < drawdown <
    whole-project kill.** The $400–500 single-day figure is a *ceiling I stay far
    away from*, never the limit I set — set the halt well below it so I never
    actually reach the pain point.

6c. **Gross exposure cap.** Total capital deployed across all open positions must
    never exceed **100% of capital** (no leverage). At small accounts where one
    share is a large fraction of the account, treat this as a hard ceiling on how
    many positions can be open at once.

---

## III. The kill switch — iron rules

7. **The kill switch is sacred. I will never manually override it to keep
   trading.** It fires for exactly the reasons I told it to (§II). If it fires,
   the system is doing its job.
8. When the kill switch fires, the protocol is: **stop, breathe, investigate —
   do not resume same-day.** Find out *why* before doing anything. Resuming is a
   next-day decision at the earliest, made calmly.
9. Resuming after a halt is a deliberate act (`resume()`), never automatic and
   never impulsive. If I find myself *wanting* to resume immediately, that is
   itself the signal to wait.
10. A halt is information, not an inconvenience. Log what happened and what I
    learned.

---

## IV. Go-live gates — no real money until ALL pass

A strategy earns real capital only after clearing **every** gate, in order. No
gate may be skipped or assumed.

- [ ] **Gate 1 — In-sample edge.** Portfolio backtest over multiple years and
      regimes shows positive, *statistically significant* results after costs:
      `is_significant == True` (PSR ≥ 0.95, ≥ ~100 trades). A pretty equity
      curve is not enough — the significance test must pass.
- [ ] **Gate 2 — Out-of-sample survival.** Edge persists on data the strategy
      was **never tuned on** (walk-forward / holdout). Degradation is expected;
      collapse is disqualifying.
- [ ] **Gate 3 — Cost honesty.** Edge survives realistic commission, slippage,
      spread, and (for shorts) borrow. If it only works at zero cost, it doesn't
      work.
- [ ] **Gate 4 — Paper matches backtest.** ≥ ~1 month paper trading where live
      fills, slippage, and timing match backtest assumptions. This validates
      *plumbing*, not edge.
- [ ] **Gate 5 — I can explain the edge** in two sentences (§I.4).

11. Paper trading is **never** evidence of edge — the sample is too small. It
    only proves the machine runs. Edge comes from Gates 1–3.
12. Most strategies will fail these gates. **That is the system working.** A
    rejected strategy saved me real money.

### IV-A. What "it works" means (definition of success)

Derived from the professional/academic consensus (López de Prado, *Advances in
Financial Machine Learning*; Bailey & López de Prado, *The Deflated Sharpe
Ratio*; Harvey, Liu & Zhu, *…and the Cross-Section of Expected Returns*). "It
works" requires **all** of the following — a rising equity curve alone is not it:

12a. **Out-of-sample, cost-adjusted Sharpe > 1.0.** Below ~0.5 is not worth
     trading. In-sample results do not count.
12b. **Statistical significance corrected for multiple testing.** Naive bar is
     t-stat > 2; because many variations get tried, the real bar for a *new* edge
     is **t > 3** / **Deflated Sharpe Ratio ≥ 0.95** (DSR discounts the Sharpe by
     the number of trials — implement before relying on this; PSR ≥ 0.95 is the
     interim proxy).
12c. **Adequate sample across regimes** — ≥ ~100 trades spanning multiple market
     environments (e.g. 2018 vol, 2020 crash, 2022 bear), not one lucky year.
12d. **Parameter stability** — neighbours of the chosen parameters are also
     profitable. If only one exact parameter set works, it is overfit.
12e. **Live matches backtest** (Gate 4) within tolerance.

12f. Conversely, **"it's broken / retire it"** = live results fall outside the
     backtest's expected band over a meaningful sample (not a few trades), or any
     of 12a–12d fails on re-validation.

---

## V. Capital deployment & scaling

13. First real money is **tuition, not investment.** Deploy an amount whose total
    loss changes nothing in my life. If a full loss of the starting stake would
    ruin my month, it is too big.
14. The purpose of the first real-money phase is to expose the one thing paper
    cannot: **how I behave when it's real.** I will behave differently. Watch for
    it.
15. **Scale up only on evidence**, never on excitement. Increase size only after
    *months* of real-money behavior matching the backtest, and only in steps.
16. Never deploy capital I would need within the trading horizon. Never trade
    rent, tuition, or emergency money. Ever.
17. ⚠️ **Personal cap (set 2026-06-23):** maximum total real capital at risk =
    **$1,000** — "tuition" to learn the operation, not a test of edge. I do not
    exceed it without a logged, slept-on decision. I understand that at $1,000,
    with $200–400/share instruments, a single share is 20–40% of the account, so
    this phase tests *plumbing and my own discipline*, not whether the strategy
    makes money (that is the backtest's job, §IV Gate 1).
17a. **Scaling ladder.** $1,000 (learn the operation) → if Gates pass and live
     matches backtest for *months*, step to ~$5,000 → then ~$8,000. Recalibrate
     §II limits at each step (table §II.6b). No step skipped; each step is a
     logged, slept-on decision.
17b. **Whole-project kill / serious-reconsideration thresholds.** A drawdown this
     deep is the master off-switch — reached only through repeated resume cycles,
     it means "stop reviewing and walk away," not "give it room."

     Provisional levels (feeling-based, to be replaced — see 17c):
     | Phase capital | Reconsider seriously | Notes |
     |---|---|---|
     | $1,000 | full loss of the $1,000 | tuition spent → stop & reassess |
     | $5,000 | ~$1,800 (≈36%) | |
     | $8,000 | ~$2,500 (≈31%) | |

17c. **Math anchor (finalise after Gate 1).** The principled threshold ≈ **2× the
     strategy's backtested maximum drawdown**, because (a) a real strategy will
     have drawdowns I must not quit on, and (b) live drawdowns run ~1.5–2× worse
     than backtest. A drawdown beyond ~2× anything seen in years of history is
     evidence the **edge has broken**, not variance. Once the portfolio backtest
     gives `max_drawdown`, replace the feeling-based numbers above with
     `2 × backtest_max_DD` (capped at what I can afford). My gut numbers imply a
     ~15–18% backtest max DD — a normal range, so they are reasonable
     placeholders until confirmed.

---

## VI. Change control — the overfitting firewall

This is the section most likely to save me from myself.

18. **Two classes of change, two different rules:**
    - **Bug fixes** (the code does something other than intended): allowed
      anytime, immediately. Fixing a broken kill switch is not "changing the
      strategy."
    - **Strategy / parameter changes** (lookback, thresholds, sizing, adding or
      removing a strategy): allowed **only** through the research process below.
19. **Strategy changes are research-driven, never result-driven.** A change is
    justified by a *re-run backtest with out-of-sample validation*, not by
    "last week lost" or "this trade should have worked."
20. **The trigger to *investigate* is not the trigger to *change*.** Investigate
    when live performance deviates from backtest beyond a pre-defined band over a
    meaningful sample. Changing in response to a handful of trades is overfitting
    to noise, and overfitting to noise is how the account dies.
21. Changes happen on a **calendar**, not on impulse: review monthly, change at
    most quarterly (see §VII). Between reviews, the system runs untouched (bugs
    excepted).
22. Every parameter I add is a degree of freedom I can overfit on and a thing
    that can silently break. **Prefer fewer knobs.** A change that removes a knob
    is worth more than one that adds one.
23. If I am tempted to change something *right now*, the answer is: **write it in
    the decision log and wait for the scheduled review.** Urgency is the enemy.

---

## VII. Review cadence

24. **Daily (≤ 5 min):** glance at the dashboard / Telegram. Confirm the system
    is alive and not halted. **Do not act on daily P&L.** Looking is allowed;
    reacting is not.
25. **Monthly review:** compare live results to backtest expectations. Record
    observations in the decision log. *Observation is not change* — most months
    end with no change.
26. **Quarterly review:** the only window where strategy/parameter changes may be
    made, and only if justified by fresh research (§VI). Re-validate out-of-
    sample. Retire underperformers (§IX).
27. Between reviews I am a spectator, not a tinkerer. The hardest discipline is
    doing nothing while watching.

---

## VIII. Drawdown protocol (decide now, not in the moment)

28. The auto-halts (§II) handle the catastrophic case. This section governs the
    *slow bleed*.
29. If real-money drawdown reaches **half** the max (≈ 2.5% from peak): no size
    increases, no new strategies, heightened attention. Investigate, don't react.
30. If the system hits the **5% max drawdown halt**: full stop. Do a complete
    review before resuming. Treat it as the system telling me a premise may be
    broken, not as bad luck to push through.
31. **I will never "trade my way out" of a drawdown** by increasing size or
    loosening limits. That instinct is precisely how accounts go to zero.

---

## IX. Strategy lifecycle

32. **Adding** a strategy requires passing all go-live gates (§IV). No
    exceptions, no "let's just try it small" without the gates.
33. **Retiring** a strategy: if a live strategy's edge degrades materially versus
    its backtest over a meaningful sample, retire it. Retirement is a normal,
    healthy act, not an admission of failure. The subtractive mindset is the
    professional one.
34. A strategy is guilty until proven innocent. The default state of any new idea
    is "off."

---

## X. The Never list (commandments)

I will **never**:

35. Override the kill switch to keep trading.
36. Add to a losing position ("averaging down").
37. Revenge trade — increase activity or size to recover a loss.
38. Change a parameter in response to recent results instead of research.
39. Skip the out-of-sample gate because a backtest looks too good to wait.
40. Deploy money I cannot afford to lose entirely.
41. Widen a risk limit during a losing period.
42. Trade a strategy whose edge I cannot explain.
43. Let code and this doctrine silently disagree.
44. Confuse paper-trading survival with proof of edge.
44a. Increase size off a **winning** streak. Euphoria is as dangerous as panic;
     size changes are research-driven only (§VI), in both directions.
44b. Enter a single-stock mean-reversion trade within ~2 days of that company's
     earnings — a gap can blow straight through the stop (a 1% stop becomes a 5%+
     loss). Flatten or stand aside around known events.

---

## XI. Decision log (mandatory)

45. Every strategy/parameter change, every go-live, every resume-after-halt, and
    every limit change is recorded — date, what changed, **why**, and the
    backtest/evidence that justified it. Use git commit messages and/or a
    `DECISIONS.md`. No change without a logged rationale.
46. "I had a feeling" is never a logged rationale. If the reason can't be written
    as evidence, the change doesn't happen.

---

## XII. Humility & known limits

47. Backtests describe the past. They are necessary, not sufficient. Expect live
    results to be worse than backtest — if I'm not pleasantly surprised, I sized
    my expectations right.
48. Edges decay. What works will eventually stop working. Monitoring for that is
    a permanent job, not a phase.
49. The market does not know I exist and owes me nothing. Profit is not deserved;
    it is extracted from a real counterparty who is also trying to win.
50. When this document and my gut disagree, **the document wins.** That is the
    entire point of writing it down while calm.

---

## XIII. Operational & regulatory constraints

51. **Pattern Day Trader (PDT).** PDT is a US FINRA rule (US margin accounts
    < $25k → max 3 day-trades per rolling 5 business days). My account is
    **Australia-based**, so PDT very likely does **not** apply — but I will
    **verify with IBKR** before relying on intraday strategies live. If it *does*
    apply: intraday strategies (ORB, Bollinger) are capped at 3 day-trades/5 days
    or disabled live; overnight strategies (momentum daily, trend 4H) are
    unaffected. **PDT never constrains backtesting** — all intraday research is
    done in the backtest regardless.
52. **Operational risk / single point of failure.** The system runs on one home
    machine (power, internet, OS updates, TWS crashes are all real risks). Open
    positions are protected by **resting bracket stops/targets on IBKR's
    servers**, which survive my system going down. If the system is down: trust
    the resting brackets; do **not** panic-manage manually unless I have
    confirmed a bracket actually failed. Keep TWS auto-restart on.
53. **Tax & records (Australia).** Being Australian changes *which* rules apply,
    not whether they matter: the ATO distinguishes **"share trader" (business
    income) from "investor" (CGT)** — frequent algo trading may make me a trader;
    the 50% CGT discount needs >12-month holds (my trades won't qualify); file a
    **W-8BEN** for the 15% US-dividend withholding under the AU–US treaty. Keep
    complete records (already logged to SQLite) and **consult an Australian
    accountant before scaling.** Not tax advice — a flag to get proper advice.

---

*Signed (by committing this file): the calm version of me, for the use of the
panicked version of me.*
