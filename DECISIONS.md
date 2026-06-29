# Decision Log

Per `TRADING_RULES.md` §XI: every go-live, halt-resume, limit change, and
strategy add/retire is recorded here with date, what was decided, **why**, and
the evidence. Newest first.

---

## 2026-06-29 (cont. 2) — Volatility risk premium: REAL edge, wrong shape & sign → rejected for this book

**Decision:** **Reject** VRP — but for a different reason than momentum/carry. Those
had *no edge*; VRP has a **genuine, persistent edge** and is still the wrong thing
to add: negatively-skewed (ruin-shaped), positively correlated to equity-crash risk
(the opposite of what this book is built to own), and not retail-accessible. Third
and final spike of the broaden-the-premia plan. **Premia hunt now closed.**

**Method** (`backtesting/vrp_research.py`; rolled short 1-month variance swap,
strike = prior month-end (VIX/100)², realised = next-month SPX variance; ^VIX +
^GSPC, 1990–2026; vol-targeted 15%; gross + net of 0.5 vol-pt/mo cost):

| | Ann. | Sharpe | Max DD | Monthly skew | Worst month |
|---|---|---|---|---|---|
| Gross | +9.1% | 0.70 | 61% | **−8.8** | −60% (Mar-2020) |
| Net (0.5 vol-pt/mo) | +7.0% | 0.57 | 61% | −8.9 | −60% |

**The edge is real** (Sharpe ~0.6 net — comparable to trend) — the only spike with
an actual signal. **But three things kill it for us:**
1. **Wrong shape:** monthly skew −8.8, 61% drawdown at a *modest* 15% vol (ruin at
   the leverage short-vol funds actually run). The 5 worst months are all crashes
   (Mar-2020 −60%, Oct/Sep-2008, Apr-2025, Aug-2011). Sharpe flatters a strategy
   that periodically detonates — PSR stays high only because the daily sample is
   huge, not because the tail is survivable.
2. **Wrong sign for THIS portfolio:** beta +0.49 / corr +0.45 to SPY — short-vol is
   *long* equity-crash risk. Our validated trend sleeve is crisis alpha (*long*
   vol; VRP corr to trend −0.09). Adding VRP would **sell the exact insurance the
   trend sleeve buys** — actively undoing the diversification that is the entire
   reason the book exists.
3. **Not accessible:** clean instrument = institutional variance swap. Retail
   proxies detonate — XIV terminated −96% in a day (2018-02-05, "Volmageddon");
   SVXY/short-VXX share the path-dependency. The ~0.5 vol-pt net premium (implied
   19.5 vs realised 19.0) is thin and cost-sensitive even before that.

**Lesson:** a high Sharpe is not an edge you want if its skew is −9 and it shorts
the tail your portfolio is designed to be long. "Real premium" ≠ "right premium."

**Premia hunt verdict (momentum → carry → VRP):** three swings, zero additions —
two had no edge, one had an edge of the wrong shape/sign. This *confirms* the
standing conclusion: the product is the **validated trend sleeve blended with
equity diversification** (2026-06-23 cont. 6), not a stack of retail "edges." Edge
is rare; manufacturing more of it on free data + retail infra is not the move.

**Running tally:** trend ✓ (the one edge), defense ✗✗, biotech untestable, OFI
unharvestable, x-sec momentum ✗, rates carry ✗, **VRP ✗ (real but wrong shape/sign,
inaccessible)**.

---

## 2026-06-29 (cont.) — Rates carry: no standalone edge → rejected

**Decision:** **Reject** carry as a buildable edge. The one carry market testable on
free data (US rates) does not beat buy-and-hold intermediate Treasuries
risk-adjusted; the apparent "alpha" is bond beta mismeasured against an equity
benchmark. Second spike of the broaden-the-premia plan; second rejection.

**Scoping (honest):** carry is the non-price return component (yield / roll-down /
term structure) — by construction *absent* from adjusted-price data. So FX-rate and
commodity-term-structure carry are **untestable on free data** (same wall as OFI).
The rates complex is the only carry market with free observable inputs (Yahoo
^IRX/^TNX/^TYX), so the test is rates-only — not the cross-asset carry the
literature validates (Koijen-Moskowitz-Pedersen-Vrugt 2018).

**Method** (`backtesting/carry_research.py`; SHY/IEF/TLT, Yahoo curve, monthly,
lagged, 10bps; two pre-committed variants):

| Strategy | Ann. | Sharpe | Max DD | Verdict |
|---|---|---|---|---|
| A. Duration timing (TLT/SHY by 10y-3m) | +4.0% | 0.37 | 44% | ≈ TLT b&h |
| B. X-sec rates carry (inv-vol) | +4.1% | 0.47 | 32% | < IEF b&h |
| **IEF buy & hold** | +3.6% | **0.55** | 24% | dominates both |
| TLT buy & hold | +3.7% | 0.33 | 48% | — |

**Why reject:** **just holding IEF (Sharpe 0.55) beats both carry variants** — the
timing/selection adds nothing risk-adjusted, only turnover. The headline "+6–7%
alpha vs SPY" is the momentum-trap inverted: scoring a *bond* strategy against an
*equity* benchmark credits the whole bond risk premium as alpha. Right benchmark
(bonds) → no edge. Plus the 2002–2026 sample is mostly the tail of the 40-yr bond
bull (a duration tailwind that broke in 2022) → forward-looking worse.

**Lesson:** benchmark a strategy against the *asset class it actually trades*, not
SPY by default. Carry's only real contribution — low equity correlation
(A/B vs SPY corr −0.30/−0.33; vs trend +0.15/+0.17) — is just the bond-equity
diversification you already get by holding bonds. No need for a "strategy."

**Running tally:** …trend ✓, defense ✗✗, biotech untestable, OFI unharvestable,
x-sec sector momentum ✗, **rates carry ✗ (< buy-and-hold IEF)**. Next & last
premium swing: **volatility risk premium**.

---

## 2026-06-29 — Cross-sectional (sector) momentum: no edge → rejected

**Decision:** **Reject** cross-sectional momentum on the sector-ETF universe. The
market-neutral version has no edge; the long-only version is equity beta wearing a
significance label. Not added. (First spike of the "broaden the premia book" plan
from 2026-06-28.)

**Thesis tested:** within a homogeneous peer group, relative winners keep winning
(Jegadeesh-Titman; Moskowitz-Grinblatt industry momentum). Distinct from the
validated *time-series* trend (relative ranking vs absolute sign). Chosen because
sector ETFs are survivorship-clean on free data → honestly testable now.

**Method** (`backtesting/xsec_momentum.py`; 11 SPDR sectors, 12-1 month momentum,
tercile long/short, dollar-neutral gross=1, monthly, weights lagged 1d, 10bps
turnover cost; single pre-committed config — no parameter sweep):

| Variant | Ann. | Sharpe | PSR | Max DD | Verdict |
|---|---|---|---|---|---|
| L/S dollar-neutral | −1.5% | **−0.15** | 22% | 39% | no edge |
| Long-only top tercile | +8.5% | 0.53 | 100% | 49% | = SPY (beta) |
| SPY buy & hold | +8.6% | 0.53 | 100% | 55% | — |

**Why it fails:** (1) ~9–11 sectors is too small/correlated a cross-section for
ranking to carry information; (2) the short leg is poison — shorting beaten-down
sectors that mean-revert bleeds the L/S book. Consistent with the literature that
industry/sector momentum is far weaker than single-stock momentum.

**Lesson (metric caveat):** the long-only variant trips `is_significant = YES`
(PSR 100%) yet has alpha≈0 vs SPY (8.5 vs 8.6%, Sharpe 0.53 vs 0.53; L/S
attribution beta −0.06, corr −0.13). `is_significant` checks PSR + sample size, NOT
beta — a long equity strategy looks "significant" because the equity premium is
real, not because the *strategy* is. Always read `factor_attribution` alongside it.
Echoes the 2026-06-23 momentum-QQQ finding (+1.2%/yr, alpha negligible).

**Diversification check:** L/S corr to the validated trend edge = **+0.44** — even
had it worked, it's partly the same momentum bet repackaged, not the clean
uncorrelated second leg the portfolio thesis wants.

**Not rejected, untestable:** cross-sectional momentum on a *broad single-stock*
universe (hundreds of names) is where the anomaly is strong — blocked by the
survivorship / point-in-time data wall (same gate as biotech). Recorded, not run.

**Running tally:** 4 technical patterns ✗, trend ✓, defense post-award ✗, defense
announcement-timing ✗, biotech = untestable, OFI = unharvestable, **x-sec sector
momentum ✗ (no edge; long-only = beta)**. Next premium to test: **carry**.

---

## 2026-06-28 — Order Flow Imbalance (Cont-Kukanov-Stoikov) rejected as alpha; parked for execution

**Decision:** **Reject** OFI as a source of directional alpha for this system. The
underlying phenomenon is real and well-established — this is *not* a "no edge"
finding like the technical patterns — but it is **structurally unharvestable** at
our data + latency, so it never reaches a backtest. Park the *execution* angle
(OFI/depth-aware order timing) as a future nice-to-have, not on the critical path.

**Paper:** Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events"
(working paper ~2010–2011; *J. Financial Econometrics*, 2014). Defines OFI as the
net signed change in size at the **best** bid/ask aggregating limit orders, market
orders and cancellations; regresses mid-price change on OFI: `ΔP = β·OFI + ε`.
Headline results: linear fit with high contemporaneous R² (~65%, beats trade
imbalance), and price impact scaling inversely with depth (`β ≈ c/D`).

**Why reject (not "no edge" — "not ours to take"):**
1. **The 65% R² is contemporaneous, not predictive** — mechanical co-movement
   (price moves *because* the book moved). Lagged/forecast R² is low single digits
   and decays in seconds. The impressive number is not a tradeable number.
2. **It's a market-maker / queue-position signal.** The edge accrues to whoever is
   at the front of the queue when imbalance builds. A price-taker who observes OFI,
   reacts, and crosses the spread captures a fraction of a tick after paying
   half-spread + fees → expected net edge ≈ 0. Adverse selection by construction.
3. **Data wall (same logic that parked biotech, cont. 4):** OFI needs L1 *message*
   data (every order/cancel/exec, µs-stamped). Not available historically on free
   data → **no honest backtest is possible.** Collectible only *prospectively* via
   IBKR L2 / tick-by-tick.
4. **Infra mismatch:** `core/broker.py` is retail ibapi. We cannot see or act on
   top-of-book fast enough; we'd trade *against* the HFTs who generate the flow.
5. **Non-stationary β** (depth follows the intraday U-shape / vol regime) — more
   conditioning, more overfit surface, for a signal we can't reach anyway.

**Honest verdict:** intellectually first-rate and empirically robust as *science*;
fails our exploitability bar (rules 8–11, §IV gates) on data + latency, before any
significance test. The genuinely useful angle for us is **execution, not alpha** —
OFI/depth-aware timing to lean against transient imbalance when entering or
rebalancing the trend / DBMF book — but that only matters once trading real size,
which the 60/40 ETF core (cont. 6) does not. Parked, not pursued.

**Running tally:** 4 technical patterns ✗, trend ✓ (modest, validated), defense
post-award ✗, defense announcement-timing ✗ (Gate-2 artifact), biotech =
untestable free (data wall), **OFI = real but unharvestable (data + latency wall)**.

---

## 2026-06-23 (cont. 8) — Defense announcement-timing FAILS Gate 2 → rejected (artifact)

**Decision:** **Reject** the defense announcement-timing signal. The Gate-1
"sell-the-news" effect was a spring-2026 regime artifact; it does not generalize
out-of-sample. Do not pursue or deploy.

**Evidence:** (1) defense.gov pagination only exposes ~4 months of history (130
pages → same 85 events, all spring 2026) — can't validate by scraping. (2) So we
ran the IDENTICAL methodology (next-open entry, abnormal vs ITA) on the multi-year
USAspending award set: **330 events, 2010–2024 → pooled drift ≈ 0** (all |t|<1.2;
20d +0.09%, t=0.26), and **per-year is pure noise** (means −2.7%…+1.9%, only 2 of
15 years reach |t|≈2, opposite signs). The −4%/t=−3.4 spring-2026 effect vanishes
in the broad sample.

**Lesson:** a pre-committed Gate-1 hit at t>3 was correctly caught as luck by Gate
2 — the discipline prevented trading an artifact. Running tally: 4 technical
patterns ✗, trend ✓ (modest, validated), defense post-award ✗, defense
announcement-timing ✗ (Gate-2 artifact), biotech = untestable free (data wall).
**One validated edge (trend) out of the lot — the honest yield of edge-hunting.**

---

## 2026-06-23 (cont. 7) — Defense announcement-timing: significant "sell-the-news" (Gate 1 pass; needs Gate 2)

**Decision:** The one pre-committed defense variant produced a **statistically
significant** signal — but a *fade*, not drift. Promising enough to take to Gate 2
(multi-year, out-of-sample) before believing it. Do NOT deploy on one year of data.

**Evidence** (`research/defense_announcements.py`; 85 new-award announcements from
defense.gov precise dates, ~1 yr; enter next-day open, abnormal return vs ITA):
overnight gap flat (+0.03%, t=0.28 → award anticipated); then significant
NEGATIVE drift: −1.84% (t=−3.27) at 5d, −2.59% (t=−3.46) at 10d, −4.12% (t=−3.44)
at 20d; only ~30% of events positive. Internally consistent across horizons.

**Interpretation:** "buy the rumor, sell the news" — names run up into anticipated
contract wins, then underperform the sector as hype buyers take profits.
Mechanism + counterparty are nameable. Pre-committed (one shot) → no
multiple-testing penalty, so t>3 is honest.

**Caveats (why it's Gate 1, not go-live):** (1) only 85 events / ~1 year / one
regime — could be a 2025–26 defense-rotation artifact; needs multi-year OOS
confirmation. (2) It's a SHORT/relative trade (short name vs long ITA) — borrow
costs + shorting friction, awkward at a small account.

**Next (if pursued):** Gate 2 — scrape several more years of announcements, confirm
the effect holds across regimes and out-of-sample; only then consider
implementation.

---

## 2026-06-23 (cont. 6) — Deploy the validated systematic core (equity + trend)

**Decision:** Deploy the validated core as a **60% equity / 40% managed-futures**
blend, implemented with ETFs, rebalanced quarterly. This is the "don't-lose"
foundation; the bot is NOT needed for it (buy + quarterly rebalance).

**Allocation:**
- 60% **broad US equity** — VTI or VOO (long-term growth engine).
- 40% **managed futures / trend** — DBMF (or KMLM) as the practical wrapper for
  the trend edge we validated (it runs trend-following across futures; same edge,
  not identical to our DIY version; fee ~0.85%).
- **Rebalance** quarterly, or when a sleeve drifts >10% from target.

**Why 60/40 (not the max-Sharpe 30/70):** the user wants long-term growth, so tilt
to equities for return, with a 40% trend sleeve as a genuine diversifier
(corr ≈ 0, crisis alpha) to cut the worst drawdowns. Backtest blend ≈ 8.6%/yr,
Sharpe ~0.78, ~31% max DD — better risk-adjusted than 100% equity and far more
survivable than equity's 55% DD. Dial toward more equity (more return, deeper DD)
or more trend (smoother) per stomach.

**Honest expectation:** ~6–9%/yr long-term with ~25–35% drawdowns. A survivable,
diversified foundation — not a get-rich plan. Start with tuition capital
(§V.17 = $1,000); recalibrate nothing (it's buy-and-hold ETFs, no per-trade risk
limits needed). US ETFs accessible via IBKR Australia (W-8BEN on file).

**Status:** plan committed. Execution (actually buying) is the user's call, when
ready.

---

## 2026-06-23 (cont. 5) — Defense post-award drift: no edge (rejected)

**Decision:** Reject the defense post-award-drift thesis. Do **not** keep firing
off defense variants hunting for significance — that is p-hacking, which our
PSR/deflated-Sharpe discipline exists to prevent. Any further variant must be
pre-committed and discounted for multiple testing.

**Evidence** (`research/defense_study.py`; 330 base awards ≥$10M across 15 defense
names, 2010–2024; cumulative abnormal return vs ITA, enter T+1 close, free
survivorship-clean data): all horizons/tiers t-stat < 1.1; best case pure-play
40d +1.09% but t=0.95 (noise); %positive ≈ 50%.

**Why:** defense awards are telegraphed/anticipated → priced in before the action
date; primes too diversified for one award to matter; action-date timing noisy vs
the actual press release. Untested variants exist (announcement-day timing via
DoD daily releases, pre-award run-up, surprise-filtered) but pursuing many invites
multiple-testing bias.

**Running tally (honesty):** 4 technical patterns → no edge; trend-following →
modest edge (diversifier); defense post-award → no edge. The lesson — genuine
edge is rare — is itself the result.

---

## 2026-06-23 (cont. 4) — Satellite data scoping: biotech → prospective; defense → backtest now

**Decision:** Run **biotech as a slow background data collector** (Path B,
prospective, free, clean) and make **defense the near-term *active* catalyst
project** — because defense can be backtested historically on free,
survivorship-clean data, while biotech cannot.

**Evidence — biotech:** openFDA gives a free 20+yr approval calendar, BUT (1)
sponsor→ticker mapping is messy/partial, (2) the tradeable small-cap universe
needs historical market cap, and (3) **survivorship bias is lethal and proven** —
yfinance returns 0 rows for delisted biotechs (Dendreon, KaloBios) vs 7,568 for a
survivor (NBIX). A free historical backtest would only see winners → fake edge.
→ Built `research/catalyst_logger.py` (logs FDA approvals prospectively, deduped,
best-effort ticker, manual-confirm). Seeded 44 clean approvals (last 12 mo).
Reality: ~44 approvals/yr, mostly manual mapping → ~1–2 yrs to a real sample.

**Evidence — defense:** USAspending.gov API is free and official; recipients are a
small, well-known, easily-mapped universe (LMT, BA, GD/Electric Boat, RTX, NOC…);
primes don't delist → **survivorship-clean with free price data.** A real
historical award→price study is feasible now. Caveats: need action/announcement
dates (not contract base dates); edge likely lives in smaller pure-play names
(LDOS, KTOS, AVAV, MRCY, CACI) where a contract is material, or surprising/large
awards — the backtest will decide.

**Next:** build the defense study — action-dated DoD awards by defense recipient
(USAspending / DoD daily announcements) → map to tickers → backtest post-award
drift on a materiality-filtered universe.

---

## 2026-06-23 (cont. 2) — Trend-following passes Gate 2 robustness (with caveats)

**Decision:** Trend-following clears Gate 2 — a robust, real edge, kept as the
validated systematic core. But it is **modest, decaying, and regime-dependent**;
size expectations accordingly and only ever deploy it blended with equities.

**Evidence** (`backtesting/trend_robustness.py`):
- **Parameter stability:** every lookback 3–15m positive, PSR ≥ 96% (best 9–12m).
  Not a knife-edge parameter → not overfit.
- **Out-of-sample:** both halves positive (Sharpe 0.64 then 0.38; PSR 98%/90%) —
  but the decay is real (crowding).
- **Regime:** 3 of 4 eras positive, but **2016–2020 was dead (Sharpe −0.05,
  PSR 46%)** — the documented managed-futures "lost decade." Trend is crisis
  alpha: great in dislocations, starves in calm/choppy markets.

**Implications / honest expectations:** future Sharpe likely ~0.3–0.4; expect
multi-year flat/losing stretches that are psychologically brutal (and *are* the
reason the edge persists). Best use: blended with equities (~30–50% equity found
best risk-adjusted, Sharpe ~0.86 vs ~0.66 for either alone, drawdown 13% vs 55%).

**Open decisions:** (a) implementation — DIY 14-position bot vs a managed-futures
ETF wrapper (e.g. DBMF) + equity ETF for a small account; (b) whether to deploy
this core now or invest effort in the satellite (biotech/hardware) research.

---

## 2026-06-23 (cont.) — Trend-following validated as the first real edge (a diversifier)

**Decision:** Adopt **diversified time-series momentum (trend following)** as the
research core — it is the first strategy to clear §IV significance. Treat it as a
**diversifier, not a standalone return-maximizer.** Next: test blending it with
equity exposure to find an allocation that beats either alone risk-adjusted.

**Evidence** (`backtesting/trend_research.py`; 14 ETF markets, 12m momentum,
inverse-vol long/short, costs, 2003–2026, no look-ahead):

| Variant | Ann. return | Sharpe | Max DD | PSR |
|---|---|---|---|---|
| Trend, unlevered | +3.6% | 0.53 | 18.4% | 99% |
| Trend, vol-targeted ~12% | +8.0% | 0.69 | 29.0% | 100% |
| SPY buy & hold | +11.2% | 0.67 | 55.2% | 100% |

vs SPY: **beta −0.01, correlation −0.03, alpha +4.0%/yr.**

**Why it's real (vs the four that failed):** PSR ~100% over a 23-year, 14-market
sample; a genuine mechanism (behavioural underreaction + crisis-alpha premium)
with a named counterparty; uncorrelated to equities.

**Honest caveats:**
- Underperforms buy-and-hold on **raw return** (3.6–8% vs 11.2%).
- Unlevered return (3.6%/yr) is below current cash; equity-like returns need
  leverage (≤3x) that a small account can't safely run.
- Sharpe 0.5–0.7 is modest (consistent with the literature).
- Value is **diversification** (corr ≈ 0, ~⅓ the drawdown), not out-returning
  stocks. Proper use: blended with equities.

**Metric fix made same day:** `is_significant` previously required ≥30 discrete
trades, which misfired on continuous strategies; it now uses PSR + observation
count when there are no discrete trades (≥30 trades still required for discrete
strategies).

---

## 2026-06-23 — Gate 1 backtest: no current strategy shows edge; do not deploy

**Decision:** Do **not** deploy real capital into any of the four current
strategies. Pivot research toward strategies with a genuine, stated edge thesis
(Path 1). None of the current strategies has cleared §IV Gate 1.

**Evidence** (`backtesting/run_backtest.py` over `data/` fetched via
`backtesting/fetch_data.py`; conservative cost model — commission + slippage +
half-spread + short borrow):

| Strategy | Period | Trades | Total | /yr | Win% | Significant? |
|---|---|---|---|---|---|---|
| Momentum (QQQ) | 10 yr | 37 | +12.5% | +1.2% | 73% | No |
| Trend (GLD/USO) | 3 yr | 55 | +0.7% | +0.2% | 42% | No |
| Bollinger (15m) | 3 mo* | 36 | −1.1% | — | 33% | No |
| ORB (AMZN 5m) | 3 mo* | 30 | +$17 | — | 53% | No |
| Full portfolio | 10 yr | 156 | +12.3% | +1.2% | — | No |

\*Bollinger/ORB limited to ~3 months by free intraday data — *unproven*, not
fully disproven; needs paid/IBKR history for a real test.

**Why:**
- The best strategy (momentum) returned **+1.2%/yr vs ~18–20%/yr for simply
  buying and holding QQQ** over the same decade, and below the risk-free rate.
- Attribution: beta 0.02, alpha ~0.9%/yr, R² 0.05 — negligible skill.
- All samples below the ≥100-trade / multi-regime significance bar.
- Consistent with theory: these are the most-published, most-arbitraged retail
  technical patterns; no edge thesis explains why they should still work.

**Caveats / notes:**
- The Sharpe figures in the run are distorted (strategies sit in cash most days,
  deflating volatility; measured vs a 5% risk-free rate). The verdict rests on
  returns/win-rate/sample size, not Sharpe. (Metric refinement is a roadmap item.)
- This is the risk framework working as intended — it prevented a live
  deployment into strategies without edge.

**Next:** Path 1 — research strategies with a real edge thesis (a *why* and a
counterparty), thesis-first, then test with this same framework. Optionally
Path 2 (better intraday data to properly test Bollinger/ORB).
