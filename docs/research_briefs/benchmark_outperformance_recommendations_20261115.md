# Why the models have not beaten SPY or QQQ, and what to change

Prepared November 15, 2026 by Claude (Fable 5.2) at the user's request. This is a
static research response in the same folder as the November 10 brief. It is **not**
the active plan and it does not replace `docs/active_edge_rebuild_plan.md` or the
handoff. Nothing here has been implemented. No code, data, model or existing
document was changed, and no model was fitted, while preparing it.

How to read the claims:

- **Measured** means a number copied from a saved repository artifact or the
  current handoff/feature audit. The source is named.
- **Arithmetic** means a calculation I made from measured numbers plus a stated
  assumption. These are approximations, not new experiment results.
- **From memory** means general quantitative-finance knowledge or published
  research that I did not re-check against sources in this session.
- **Proposal** means a design choice for you to accept, change or reject.

## 1. Short answer

The project has not beaten SPY or QQQ for four reasons that the saved evidence
supports directly, and none of them is a software defect:

1. **The inputs contain almost no information about which stock will do better
   over the next ten sessions.** The best measured ten-session result orders
   stocks barely better than chance. Its expected gain per trade is smaller than
   the 20 basis point trading cost.
2. **The comparison is harder than it looks.** The models pick an equally weighted
   basket from S&P 500 members, but SPY and QQQ are weighted by company size. In
   the 2022 to May 2024 scoring period the average eligible stock trailed SPY by
   about 7 percentage points over a year. A stock picker has to recover that
   deficit before it shows any gain. Nothing in the current design addresses it.
3. **There is too little history to learn or to prove anything.** Every modeled
   decision starts on 2019-07-09 because news coverage starts there. For a one-year
   forecast that is about four non-overlapping periods. The acceptance rule used in
   August would reject a genuinely good strategy about five times out of six.
4. **The research loop is too slow to find signal.** About one third of outcomes
   are marked unknown because each dividend and corporate action must be proven
   from documents, so most portfolio results cannot be computed at all. In
   October a twelve-policy backtest campaign ran for six days and produced zero
   usable portfolio comparisons. Roughly half of all commits record, freeze or
   close something rather than test a prediction idea.

The remembered "0.67" models were not stronger stock pickers. They predicted which
stocks make large moves in either direction, which is mostly a ranking of
volatility, not of direction (section 3.4).

My main recommendation is to stop treating this as a model-tuning problem and
change five things together: the target (rank stocks against each other, monthly
horizon), the labels (total-return adjusted prices so nearly every row has an
outcome), the training set (longer history and more stocks for the price and
financial inputs), the features (documented return predictors, especially
earnings-announcement reaction and financial-statement measures, instead of chart
indicators and news counts), and the evaluation policy (always fully invested and
measured as a tilt away from the benchmark). Section 6 orders this as four small
steps, each with a rule for stopping.

An honest expectation, **from memory** and **arithmetic**: a well-built long-only
selection model on large US stocks using public prices, filings and news can
plausibly add about 1 to 3 percentage points a year over its benchmark after
costs, and will still trail it in roughly four years out of ten. Beating QQQ with
stocks drawn from the S&P 500 is mostly a bet on large technology companies, not a
test of prediction skill. No design in this document guarantees outperformance.

## 2. What I inspected, including the newest branch and `main`

- `main` at `aa8def5` (November 13) is the newest ref in the repository, local and
  remote. Working tree was clean.
- The newest branch other than `main` is `codex/memory-limit-target-names-20261107`
  (November 7). It has **0** commits that are not already in `main`. The same is
  true of the three other November 6 to 7 investment branches,
  `codex/model-evidence-pipeline-20261105`, `codex/v1-trained-return-api` and
  `codex/v1-canonical-cleanup`.
- Only two code changes exist on a branch and not in `main`:
  `46c5309` on `codex/swing-one-atr-stop` (stop-loss at one average true range,
  target at three) and `0ce8edc` on `codex/gpu-xgboost-training` (optional CUDA
  training). `archive/merge-checkout-20261031` holds three housekeeping-rule
  commits. Neither code change affects the diagnosis below. I recommend leaving
  both unmerged: a tighter stop cannot help when the entries carry no signal, and
  training speed is not the bottleneck (the November 11 ranking run fitted twelve
  models in about two and a half minutes).
- Documents read: `AGENTS.md`; the leading November sections of the handoff and
  feature audit; the swing sections of the feature audit; the November 10 research
  brief; model cards; `configs/edge_rebuild_strategy_contract.toml`.
- Saved results read: `models/edge_rebuild_swing_candidate_20260802_v2` and `_v3`
  `evaluation.json`; the three July `*.joblib.manifest.json` files;
  `docs/evidence/ks3_*` and `ks4_*`.
- Code read: swing feature groups (`swing/features/panel.py`,
  `cross_sectional.py`), labels (`swing/labels/barrier_and_rank.py`), selection
  (`swing/selection.py`), estimators (`swing/training/return_estimators.py`).

I did not rerun any experiment, load any model or recompute any metric from data.
The machine had 1.6 GB of 15.7 GB memory free during this review, above the 90%
ceiling, with no project job running.

## 3. What the saved evidence says

Terms used below. **Rank correlation** is how closely the model's ordering of
stocks on one date matches the ordering of their later returns: 0 means no
relationship, 1 means a perfect ordering. **Excess return** is the stock's return
minus the benchmark's return over the same interval. **AUC** is the chance that
the model scores a randomly chosen winner above a randomly chosen loser: 0.50 is
a coin flip.

### 3.1 Ten-session swing forecasts: no usable ordering (measured)

Source: feature audit, "Technical Return Experiment Acceptance" and "Relationship
Return Results". Decisions are daily S&P 500 member rows from 2019-07-09 to
2024-05-28: 586,305 decision rows, 545 securities, 1,231 sessions. Scored
out-of-fold predictions: 289,802 for later dates, of which 187,432 have a known
outcome and 102,370 (35.3%) do not.

| Inputs and learner | Rank correlation, later dates | Rank correlation, stocks never trained on | Squared error versus always predicting zero excess |
| --- | ---: | ---: | --- |
| 120 price and volume inputs, linear | 0.0045 | 0.0070 | 1.8% worse |
| 120 price and volume inputs, boosted trees | 0.0147 | 0.0003 | 2.0% worse |
| 124 inputs (adds long momentum), linear | 0.0007 | 0.0055 | worse |
| 124 inputs (adds long momentum), boosted trees | 0.0063 | -0.0135 | worse |

The August classification candidates agree. The four tree-ranking candidates each
bought the 25 highest-scored stocks every day for 252 sessions (about 6,300
simulated trades each). Their average excess return over SPY per trade was between
-0.17% and -0.01%, with AUC between 0.53 and 0.55. Source:
`models/edge_rebuild_swing_candidate_20260802_v3/evaluation.json`.

**Arithmetic: how much ordering skill the ten-session product needs.** The
spread of ten-session excess returns across stocks is about 5.1% (square root of
the measured zero-forecast squared error, 0.00256). If the top 5% of stocks are
bought, their expected advantage over the average stock is roughly
rank correlation × 5.1% × 2.06, where 2.06 is the average standard score of the
top 5% of a bell-shaped distribution. So:

| Rank correlation | Expected advantage per ten-session trade | Meaning |
| ---: | ---: | --- |
| 0.015 (best measured) | about 0.15% | Less than the 0.20% trading cost |
| 0.019 | about 0.20% | Pays the cost, earns nothing |
| 0.038 | about 0.40% | About 5% a year after costs at 25 trades a year |

Real returns have fatter tails than a bell curve, so treat these as rough sizes.
The conclusion does not depend on precision: the measured skill is between
one-third and one-tenth of the 0.038 needed, and near zero on unseen stocks.

### 3.2 Quarterly and annual investment forecasts: some ordering, wrong level (measured)

Source: handoff, "Completed stock selection ranking experiment November 11" and
the November 10 sections. 29 monthly decision dates from 2022 to May 2024; 11,091
scored stock-date rows for later dates, of which 7,126 annual outcomes are known
and 3,965 (35.8%) are not.

| Finding | Number |
| --- | --- |
| Average forecast of annual excess over SPY (boosted trees, all inputs) | +7.50% |
| Average actual annual excess over SPY, same known rows | -7.11% |
| Squared error versus predicting zero, annual | 28% to 56% worse, depending on inputs and learner |
| Adding news and financial inputs | Worse in 7 of 8 squared-error comparisons |
| Rank correlation, annual, later dates, best variant (rank target) | 0.101 |
| Rank correlation, annual, unseen stocks, best variant | 0.053 |
| Rank correlation, quarterly, later dates / unseen stocks | 0.054 / 0.001 |
| Monthly top-tenth baskets with every outcome known, later dates | 0 of 29, for every variant |
| Annual top-tenth baskets on unseen stocks after the five-stock exclusion | 7 of 29 complete: 1 beat SPY, 6 trailed |

**Arithmetic: why annual ordering skill of 0.10 still trails SPY.** The spread
of annual excess returns across stocks is about 28% (the brief's decomposition
implies a zero-forecast squared error near 0.083). A top-tenth basket with rank
correlation 0.10 is expected to beat the average stock by about
0.10 × 28% × 1.75 ≈ 4.9 percentage points. But the average stock trailed SPY by
7.1 points. Expected result: about 2 points behind SPY, which matches the
measured complete baskets (+2.3, -7.2, -7.1 before the exclusion). Breaking even
with SPY in that period needed rank correlation near 0.15.

This is the single most important number in the evidence. The annual model does
have some real ordering ability on later dates. It loses to SPY because an equally
weighted basket of ordinary S&P 500 members was the wrong thing to compare with a
size-weighted index during a period when a few very large companies drove the
index.

### 3.3 Policy candidates that looked profitable were not distinguishable from luck (measured)

In August the logistic candidate selected 648 trades in 252 sessions, averaged
+0.73% excess over SPY per trade and compounded +19.1%. On stocks it had not
trained on it selected 48 trades and averaged -2.39%. Its AUC was 0.512. Averaged
over all 252 sessions, including days with no entry, its excess over SPY was
+0.51%, and the range of plausible values (a 95% interval from resampling
20-session blocks) ran from -0.18% to +1.37%. That range includes zero, so the
candidate was rejected. The rejection was reasonable, but see section 4.6: the same rule would
also reject most genuinely good strategies.

### 3.4 The remembered 0.67 models predicted volatility, not direction (measured)

Source: `models/sp500_6m_next_day_*_v2_20260708_candidate.joblib.manifest.json`.
All three were trained on six months of data (2026-01-08 to 2026-07-06, 499
tickers).

| Target | AUC | How often the event happens | Precision when the model says yes |
| --- | ---: | ---: | ---: |
| Next-day big gain | 0.666 | 5.7% of stock-days | 21.9%, at 1.2% of events caught |
| Next-day big loss | 0.637 | 5.0% | 5.9%, at 0.04% of events caught |
| Next-week big gain | 0.713 | 5.8% | 18.4%, at 1.9% of events caught |

The same inputs predict big gains and big losses about equally well. That is what
a model looks like when it has learned which stocks are volatile: a stock that
often moves 3% in a day has more big up days and more big down days. Volatility
is persistent and easy to predict. Direction is not. The August intraday model
with AUC 0.668 confirms it: its selected trades averaged about -0.10% after costs.
These models should not be wired to the API as a "strong V1", and AUC on
big-move or target-before-stop labels should not be used again as evidence of
stock-picking ability.

## 4. Why: seven causes, each with its effect and remedy

### 4.1 The inputs are chart indicators and news counts on the most heavily analyzed stocks

**Observed.** The 120 swing inputs are returns over 1 to 60 days, RSI, MACD,
Bollinger position, moving-average distances, on-balance volume, and their ranks
against peers (`swing/features/panel.py`). The news inputs are article counts,
average FinBERT sentiment, source counts and story age over one- and three-day
windows. The financial inputs are six ratios, complete for 6,017 of 22,346
financial decisions (26.9%).

**Effect.** These are the inputs every participant already watches, applied to
the 500 most closely followed companies, entered at the next day's open. Public
news about a large company is reflected in its price within minutes to hours, so
a count of yesterday's articles says little about the next ten sessions. The
measured rank correlations near zero are what this predicts.

**What is missing.** The repository has no input for the return predictors with
the longest published track record (**from memory**): reaction to the latest
earnings announcement, change in reported earnings versus the same quarter a year
ago, price-scaled valuation (earnings, cash flow or book value divided by price),
profitability relative to assets, accounting accruals, growth in total assets,
net share issuance, closeness to the 52-week high, stock-specific volatility, and
one-month reversal. The post-earnings strategy was recorded as blocked in July
because analyst-estimate data was incomplete; section 5.3 shows how to build it
from data already held.

### 4.2 The benchmark is size-weighted and the baskets are not

**Observed.** Average actual annual excess over SPY for eligible stocks was
-7.11% (section 3.2). The frozen selection policy caps any sector at 20% of the
basket, 33.3% at most (`configs/edge_rebuild_strategy_contract.toml`), and scores
each stock relative to its sector.

**Effect.** **From memory**, information technology was roughly 30% of SPY and
about half of QQQ in 2024. A policy that cannot hold more than 20% in one sector
is forced to hold less of the sector that drove both benchmarks. Combined with
equal weighting, the basket starts each period several points behind before any
prediction is made. This is a property of the comparison, not of the model.

**Remedy.** Measure skill and product separately (section 5.5): skill is the
selected basket versus the equal-weighted average of all eligible stocks; the
product is a portfolio that starts from the benchmark's own weights and tilts
toward higher-scored stocks.

### 4.3 The target asked for return levels, and the level changed between training and scoring

**Observed.** The regression target was each stock's excess over SPY. Training
years 2019 to 2021 taught the model that the average stock beats SPY, so it
forecast +7.5%. In 2022 to 2024 the average stock lost 7.1%. Between 21% and 27%
of the annual squared error is this date-level miss. When the November 11
experiment replaced the target with each stock's rank among same-date peers, rank
correlation rose from 0.078 to 0.101 on later dates and from 0.003 to 0.053 on
unseen stocks.

**Effect and remedy.** Whether stocks in general beat the index next year is a
market-breadth question that five years of data cannot answer. Which stock beats
its peers is the question a selection model can learn. Use the rank target
everywhere and stop reporting squared error against zero as a headline.

### 4.4 Five years of history is too short

**Observed.** `AGENTS.md` section 3.10 sets the first permitted decision date to
2019-07-09 because issuer news starts there. The first annual training fold had
17 months and 2,678 rows; the encoded input had 430 columns (215 inputs plus one
missing-value flag each).

**Effect.** Price and financial-statement inputs do not need news, yet they are
limited to the news period. Annual outcomes from monthly decisions overlap almost
completely, so five years holds about four independent annual observations.
Boosted trees with 430 columns and a few thousand overlapping rows mostly
memorize the training years. Adding inputs made results worse, which is the
expected sign of this.

**Remedy.** Train a base model on price and financial inputs over as many years
as can be obtained, and add news as a second, smaller model fitted only on the
news period (section 5.2).

### 4.5 One third of outcomes are unknown, and not at random

**Observed.** 35.3% of scored ten-session outcomes and 35.8% of scored annual
outcomes are unknown. The listed reasons are dividend currency not proven (378
holding keys), missing corporate-action dates (219), merger terms (198), cash
entitlement (198) and similar. November 11 to 13 recovered 213 of 1,199 unknown
selected holdings. No later-date monthly basket is complete.

**Effect.** Dividend payers, foreign-domiciled companies and merger participants
are removed far more often than other stocks, so the known rows are a biased
sample. More practically, the question "did the selected basket beat SPY" cannot
be answered for any later-date month, for any model.

**Arithmetic.** A typical quarterly dividend is about 0.3% to 0.5% of the price.
The spread of annual returns across stocks is about 28%. An error of half a
percent in a few outcomes cannot change which stocks rank high. Discarding 36% of
outcomes can.

**Remedy.** Use the provider's split- and dividend-adjusted daily prices for
training and ranking labels, where nearly every row has a value. Keep the
document-proven accounting as a check on a sample, and for any model that is
later proposed for release (section 5.1).

### 4.6 The acceptance rule cannot recognize a good strategy

**Observed.** The August gate required that the low end of the 95% plausible
range be above zero for daily portfolio return, for doubled costs and for excess
over the worst benchmark, all within one 252-session window, on both later dates
and unseen stocks. 24 swing specialist candidates, 22 intraday candidates and 6
swing candidates were rejected under gates of this kind.

**Arithmetic.** Call a strategy "very good" if its yearly excess return equals
its own yearly variability (information ratio 1.0; **from memory**, long-only
stock funds that sustain 0.5 are considered excellent). With one year of data,
the chance that such a strategy shows a 95% range entirely above zero is about
17%. For 0.5 it is about 7%. Requiring several such tests at once lowers it
further. To confirm a 0.5 strategy at that standard takes about 16 years.

**Effect.** Every candidate fails, so failures carry no information, and effort
moves to the next target. The target has changed at least six times since July
(next-day big move, five-session net positive, ten-session target-before-stop,
ten-session excess regression, quarterly and annual regression, within-date
rank). No single scoreboard has been kept across them, so progress cannot be
measured.

**Remedy.** Section 5.6: judge on the consistency of rank correlation across
many years, keep one fixed scoreboard, and count every trial.

### 4.7 Each experiment costs days, mostly outside modeling

**Observed.**
- 884 commits since June. By keyword in the subject line (a rough count with
  overlap), 442 mention record, freeze, close, checkpoint, handoff or docs; 138
  mention train, fit, model, experiment, feature, evaluate or compare; 126 mention
  dividend, accounting, corporate action, identity, currency or outcome; 105
  mention memory, storage, resume, retry or cleanup.
- The active plan is 990,982 bytes and the handoff 685,788 bytes, against a
  covenant rule to keep documentation small. Source is 142,143 lines in 477 files;
  tests are 99,328 lines.
- The October twelve-policy campaign ended on October 25 with ten memory failures,
  two unavailable valuations and zero funded comparisons. The rerun stopped on
  October 26 at seven of twelve with zero admitted results.
- The machine has 15.7 GB of memory. Other applications held enough that the 90%
  system ceiling was exceeded with no project job running when I checked.

**Effect.** Finding a weak signal in noisy data needs many cheap, comparable
trials. The current process gives each trial the cost of a release: frozen scope,
immutable export, independent audit, document closure. Those controls are right
for something that will be served. Applied to exploration, they have produced
about a dozen distinct fitted ideas in five months.

**Remedy.** Two lanes with different rules (section 5.7).

## 5. Recommended design (proposal)

### 5.1 Objective and target

- **Target:** each stock's forward total return, converted to its rank among
  eligible stocks on the same date, with a second version ranked within sector.
  Total return comes from split- and dividend-adjusted daily prices.
- **Horizons:** fit 5, 21 and 63 sessions together. Make **21 sessions** the
  primary research horizon and keep 10 sessions as a scored secondary so the
  swing product is not silently replaced. Reason (**arithmetic**): at 21 sessions
  the spread of returns is about 7.3%, so the 20 basis point cost is covered at a
  rank correlation near 0.013 instead of 0.019, and the slow predictors in
  section 5.3 act over weeks, not days.
- **Drop** squared error against zero, AUC on big-move labels and
  target-before-stop classification as selection evidence. Keep the
  target/stop exit only as one tested policy variant.
- **Label tiers:** research label from adjusted prices for every row; the
  existing document-proven holding accounting only for a random audit sample (to
  measure how far the two differ) and for a candidate that reaches release review.

### 5.2 Training set

- **History.** Separate the two clocks. News-dependent inputs keep the
  2019-07-09 start. Price and financial-statement inputs use every year available.
  The repository already holds daily bars from 2018-05-29. **From memory**,
  Alpaca's history reaches back to 2016 and SEC structured financial data to about
  2009 to 2011; both need checking. Reaching 15 or more years needs a purchased
  price history that includes delisted companies. That is a spending decision
  (section 7).
- **Universe.** Train on a wider set than the S&P 500, for example the 1,000 to
  1,500 most traded US common stocks each month, selected from trailing dollar
  volume so no index-membership file is needed. More stocks per date gives more
  independent comparisons, and published predictors are stronger outside the
  largest companies (**from memory**). Scoring and selection can still be limited
  to S&P 500 or Nasdaq-100 members. This requires bars for companies that were
  later delisted; if those are missing, the wider set would overstate returns and
  must not be used.
- **Two-stage model.** Stage one: price and financial inputs, long history, wide
  universe. Stage two: news inputs fitted on what stage one failed to explain,
  2019 onward. This keeps each model's source set constant through its own
  history, as `AGENTS.md` section 3.9 requires, and lets news be judged by what it
  adds.
- **Weights and overlap.** Keep equal total weight per date and the existing
  purge and embargo. Sample decisions weekly for training instead of daily: daily
  rows with a 21-session outcome are near-copies of each other.

### 5.3 Features

Build each as a plain function of the panel, then rank it across stocks on each
date. All of the following can be made with data already collected unless marked.

| Group | Inputs | Data needed |
| --- | --- | --- |
| Trend | Return from 12 months ago to 1 month ago; from 6 months ago to 1 month ago; the same after removing market and sector movement; closeness to the 52-week high; sector-level trend | Daily adjusted bars |
| Reversal | Last 5 sessions and last 21 sessions return, raw and after removing sector movement | Daily adjusted bars |
| Risk | Stock-specific volatility over 60 sessions; sensitivity to the market; largest single-day gain in the last month | Daily bars, SPY |
| Liquidity | Dollar volume, price impact per dollar traded, change in share turnover | Daily bars |
| Earnings reaction | Stock return minus market return over the three sessions around the latest earnings release; abnormal volume on that day; sessions since the release | Daily bars plus the release time. The release is the SEC 8-K with Item 2.02, and the repository already stores SEC acceptance times |
| Earnings change | Latest quarterly earnings per share minus the same quarter a year earlier, divided by its usual variability; the same for revenue | SEC company facts, already collected, dated by filing acceptance |
| Value | Earnings, free cash flow, sales and book value, each divided by market value | SEC company facts; shares outstanding × price |
| Quality | Gross profit divided by assets; accruals (earnings not backed by cash); growth in total assets; net shares issued or bought back; debt relative to assets | SEC company facts |
| News reaction | For each dated story: the stock's market-adjusted move and abnormal volume in the first session after it, kept by news category; unusual article volume versus the stock's own 60-day norm | Existing news archive and bars |
| Analyst estimate changes | Change in the average earnings forecast; surprise versus forecast | **Not held.** Paid data. Often the strongest single group (**from memory**) |

Notes:

- The earnings-reaction group needs no analyst estimates: the market's own
  response on the announcement day is the surprise measure. This reopens the
  post-earnings strategy that was marked blocked.
- For news, measure how the price responded instead of reviewing what each
  article says. This replaces most of the manual source-review work for the
  reaction profile. If a language model is used to label news categories, note
  that a model trained after the event may already know what happened next; limit
  it to naming the category and never ask it for an outcome judgment.
- Raise financial coverage above the current 26.9% by accepting the standard
  alternative tag names for the same line item, recording which tag was used.
- Market-wide inputs (index trend, volatility level) are identical for every
  stock on a date, so they cannot help rank stocks on their own. Use them only as
  switches that change how other inputs are weighted, and only after the base
  model works.
- Remove the per-column missing-value flags for boosted trees (the library
  handles missing values itself); this halves the column count.

### 5.4 Model

- **First model: an equal-weighted average of the ranked inputs above**, with
  signs fixed in advance from published research. No fitting at all. This is the
  baseline every learned model must beat.
- **Second: ridge regression on ranked inputs.** Third: boosted trees with strong
  limits (depth 3, at least several hundred rows per leaf, half of columns sampled
  per tree, a few hundred trees, averaged over five random seeds).
- No parameter search until a fixed setting has shown a result. When a search is
  run, it uses only the training folds.
- Outputs for the API: expected rank position, a rank-based expected excess with
  its measured error band, and input availability. Sizing stays in `trading_flow`.

### 5.5 Selection and trading policy used for evaluation

Execution and sizing belong to `trading_flow`. The research evaluation still has
to assume a policy, and the current one handicaps the result.

- **Always fully invested.** Money not in a selected stock earns the benchmark,
  not zero. A long-only policy that waits in cash is betting against the index
  whenever it waits.
- **Two reports, never mixed.**
  1. *Skill:* the top tenth by score, equally weighted, minus the equal-weighted
     average of all eligible stocks. This isolates prediction from index
     composition.
  2. *Product:* start from each stock's weight in the benchmark and scale it up or
     down by score, within a limit per stock and per sector **relative to the
     benchmark's own weight** (for example ±2 points per stock, ±5 per sector).
     Report this against SPY for the S&P 500 set and against QQQ for a separate
     Nasdaq-100 set. Benchmark weights need market value, which is shares
     outstanding × price.
- **Replace absolute sector caps** (20% and 33.3%) with benchmark-relative
  limits.
- **Hold rules.** Rebalance weekly. Buy a stock when it enters the top tenth; sell
  only when it leaves the top third. This buffer cuts trading roughly in half
  compared with replacing the basket every period (**from memory**).
- **Stops and targets.** Evaluate holding to the rebalance as the default.
  Compare the 1.5 and 3 average-true-range exit, and the one-range stop on the
  unmerged branch, as variants on the same entries. Do not assume a stop helps:
  with the conservative stop-first rule it converts ordinary noise into realized
  losses.
- **No market timing.** Do not add rules that move to cash on a market signal.
- **Costs.** Keep 20 basis points as the stress case. Consider 10 as the base
  case for S&P 500 members; that is a change to a frozen number and is your
  decision.

### 5.6 Validation and acceptance rules

One scoreboard, computed the same way for every trial:

- average rank correlation per date at 5, 10, 21 and 63 sessions, with a
  significance figure that allows for overlapping outcomes;
- the same by calendar year, and the share of years that are positive;
- skill report and product report from section 5.5, after costs, with turnover;
- results on later dates and on stocks never trained on, shown side by side;
- the count of trials run so far against this scoreboard.

Proposed acceptance rule for a base model, fixed before any fit:

1. average 21-session rank correlation of at least 0.03 on later dates across the
   whole walk-forward history, at least three standard errors above zero;
2. positive in at least 70% of calendar years;
3. unseen-stock rank correlation at least half of the later-date figure;
4. skill report positive after 20 basis point costs;
5. beats the equal-weighted average of inputs from section 5.4.

Three standard errors instead of two is the allowance for having tried many ideas
(**from memory**: the standard recommendation for return predictors). An added
input group, such as news, is accepted when the paired improvement in rank
correlation is at least two standard errors above zero and positive in most
years.

Keep the locked swing period (2025-07-01 to 2026-06-30, recorded as never read)
sealed until one candidate is frozen. The 2024-06-12 to 2025-06-13 window was
already read by the August evaluation, so it is development evidence. Because
any backtest on inspected history is development evidence,
start recording live predictions now: publish the section 5.4 equal-weighted
score every day from the running collector and let outcomes mature. Forward
evidence takes calendar time that cannot be recovered later.

### 5.7 Architecture and components

**Keep unchanged:** collectors, canonical bars, point-in-time index membership,
SEC acceptance clocks, the news archive, purged time-ordered validation, the
shared memory lease, API V1 and the serving and promotion controls.

**Add a research lane, small and separate:**

| Component | What it is |
| --- | --- |
| Research panel | One table per field, dates down and securities across, in `float32`: adjusted open, high, low, close, volume, total-return index, sector, market value, eligibility flag. About 1,800 dates × 1,500 securities is under 11 MB per field. Built once from canonical data, addressed by content hash |
| Feature library | Pure functions from the panel to one ranked feature each. One shared test proves that truncating the panel after any date leaves earlier values unchanged |
| Fast evaluator | Matrix arithmetic that produces the whole section 5.6 scoreboard for one score table in seconds, not a per-lot ledger |
| Trial log | One line per run: code commit, panel hash, settings, scoreboard. This replaces narrative entries in the plan and handoff for exploratory runs |
| Graduation step | Only a candidate that passes the section 5.6 rule enters the existing governed path: frozen scope, document-proven accounting on its selected holdings, independent audit, forward record |

The fast evaluator also removes the memory failures: the October campaign replayed
about 17,700 selections through per-lot ledgers holding 717 daily snapshots each.

**Pause:** recovery of the remaining unknown holding outcomes as a condition for
research; the twelve-policy campaign; the 209/130-column comparison of news and
financial inputs on the annual target (its answer would not change the design);
wiring any July or August model to the API as "V1".

## 6. Ordered steps, each with a stop rule

Effort figures are my estimates.

**Step 0. Decisions (section 7).** No code.

**Step 1. Panel, evaluator and a check that known predictors appear in this
data.** About one week. Build the research panel from existing 2018 to 2024
S&P 500 data, the fast evaluator, and about twelve price-based inputs from
section 5.3. Report each one's rank correlation at 5, 10, 21 and 63 sessions.
*Stop rule:* if textbook predictors such as one-month reversal and low
stock-specific volatility show no relationship at all here, inspect the panel and
labels before anything else. If the equal-weighted combination is at or above
0.02 at 21 sessions, the project already has more ordering skill than any fitted
swing model has shown, without machine learning.

**Step 2. Financial and earnings-reaction inputs, longer history, wider
universe.** Two to three weeks. Add the earnings, value and quality groups; extend
history and stocks as far as the step 0 decisions allow; fit the three models of
section 5.4. *Stop rule:* the section 5.6 acceptance rule. If the best model
stays below 0.02 on extended data, conclude that public price and filing data on
large US stocks will not support this product, and choose between buying
analyst-estimate data, accepting a benchmark-tilt product with a small expected
gain, or stopping.

**Step 3. News as a second stage.** One to two weeks. Add the news-reaction group
on what the base model leaves unexplained. *Stop rule:* keep news only if the
paired improvement passes section 5.6; otherwise leave it as a displayed overlay.

**Step 4. Product evaluation and forward record.** Freeze one candidate, read the
sealed period once, run the governed path on it, and compare the benchmark-tilt
portfolios with SPY and QQQ. The forward daily record continues from step 1.

## 7. Decisions needed from you

Each of these conflicts with a current rule, so none should be started without
your explicit answer.

| # | Decision | Rule it changes | My recommendation |
| --- | --- | --- | --- |
| 1 | Use adjusted-price total returns as research labels for every row; keep document-proven accounting for audit samples and release candidates | `AGENTS.md` 4.3 (unknown stays unknown) | Yes. Largest single unblocker |
| 2 | A research lane where exploratory fits need a trial-log line, not a frozen checkpoint, independent audit and document closure | 3.1, 3.4, 3.7, 6 | Yes, with the lease and memory rules kept |
| 3 | Rank target; 21 sessions primary, 10 sessions kept as scored secondary | Frozen ten-session target (October 8 note) | Yes |
| 4 | Decisions before 2019-07-09 for a base model that uses no news | 3.10 | Yes |
| 5 | Train on stocks beyond the S&P 500 | Universe scope in the strategy contract | Yes, only if delisted-company bars are confirmed |
| 6 | Evaluate as a benchmark tilt with benchmark-relative sector limits | Frozen sector caps; sizing boundary with `trading_flow` | Yes for evaluation; sizing stays in `trading_flow` |
| 7 | New acceptance rule (section 5.6) | Promotion gates in the plan and training protocol | Yes |
| 8 | Buy long price history with delisted companies, and analyst-estimate history | Source set in 3.10; cost | Decide after step 1 |
| 9 | Close the Astra six-specification, twelve-policy experiment as unfinished | October 4 confirmation | Yes. Its remaining steps cannot change the conclusion that ten-session inputs lack signal |
| 10 | Base cost 10 basis points with 20 as stress | Frozen 20 basis points | Optional |

## 8. What I recommend against

- More learners, deeper trees, neural networks or a parameter search on the
  current inputs. The limit is information in the inputs, not model capacity.
- Calibrating or shrinking the annual forecasts further. It lowers squared error
  without changing which stocks are selected.
- Excluding more stocks after seeing their returns.
- Rebuilding collection, schemas, naming or provenance.
- Reviving day-trading, or treating any AUC on a big-move label as progress.
- Merging the one-range stop or the GPU option now.
- Describing any result on 2019 to 2024 as proof. It is development evidence.

## 9. What this would and would not establish

If steps 1 and 2 succeed, the project would have a measured, repeatable ordering
of stocks with a known error band, and a portfolio definition whose difference
from SPY comes from that ordering instead of from index composition. That is the
first point at which "beats SPY" becomes a testable statement.

It would not establish profitability. **Arithmetic**, assuming a 21-session rank
correlation of 0.03, a 7.3% spread and the top fifth held: about 0.03 × 7.3% × 1.4
≈ 0.3% a month, near 4% a year before costs against the equal-weighted average,
and perhaps 2 to 3% after costs. A benchmark tilt captures only part of that.
With yearly variability of the difference around 4 to 6 points, a losing year
against SPY remains likely about two years in five, and confirming the edge from
forward results alone takes several years. Beating QQQ is achievable only as a
tilt within Nasdaq-100 members, or by accepting a large concentration in
technology companies, which is an allocation choice and not a prediction.

## 10. Evidence locators and limits

- Swing ten-session results: `docs/reviews/feature_engineering_audit_20260801.md`,
  sections "Relationship Return Results" and "Technical Return Experiment
  Acceptance".
- Investment results: the consolidated historical evidence in section 11 below.
- August candidates: `models/edge_rebuild_swing_candidate_20260802_v3/evaluation.json`.
- July classifiers: the three `models/sp500_6m_*_v2_20260708_candidate.joblib.manifest.json`.
- Policy constants: `configs/edge_rebuild_strategy_contract.toml`.

Limits of this review: I read summaries and saved metrics, not every artifact. I
did not verify the 35% unknown-outcome rows are concentrated in dividend payers
beyond the listed reason counts. Sector weights of SPY and QQQ, data-vendor
history ranges, published-research claims and typical information ratios are from
memory and should be checked before they drive a purchase. The commit keyword
counts are approximate. The arithmetic uses a bell-curve approximation and rounded
inputs; it shows sizes, not precise forecasts.

## 11. Historical investment evidence (consolidated November 22)

This replaces the November 10 external-review request; its original prose remains
in Git at `cb01c63`. It is historical evidence, not an active experiment or a
current acceptance decision. The active plan/handoff and feature audit govern
current work. No source data, model, audit receipt or evidence hash was changed.

- The investment experiment used 252-session primary and 63-session diagnostic
  targets, separately from swing research. It reused the 209-column nightly
  snapshot and appended six financial measures. Compared profiles contained 124
  technical/relationship columns and 215 full columns (124 + 85 news + 6 financial).
- Completed fitting comprised 48 validation fits and eight final research models:
  two horizons, two profiles, Ridge/XGBoost, temporal/security-transfer validation
  and three score years. Fold-local preprocessing used training medians,
  missingness indicators and Ridge scaling. Dates received equal total weight;
  outcomes had to mature before the 21-session pre-test boundary.
- All sixteen aggregate horizon/profile/learner/scope MSEs and MAEs were worse
  than zero SPY-excess prediction. Full inputs worsened seven of eight paired
  MSEs and every paired MAE. Annual temporal XGBoost MSE was 27.81% above baseline
  for technical inputs and 33.81% above baseline for full inputs; quarterly values
  were 5.22% and 4.85%. These are prediction errors, not portfolio returns.
- Each temporal specification scored 11,091 rows over 29 monthly dates. Annual
  known outcomes numbered 7,126; quarterly known outcomes 7,862. Every temporal
  selected top-decile cohort had an unknown outcome, so none of the 29 monthly
  selected-cohort returns was complete. Selection preceded outcome availability.
- The joined dataset retained 74,458 horizon rows. Its source-validated outcomes
  numbered 27,189. These units and denominators differ from the later accounting
  recovery counts in the active plan; do not substitute one for another.
- The completed diagnostic attributed 21–27% of annual temporal MSE to date-level
  mean errors and 73–79% to within-date errors. Full inputs changed forecasts too
  much relative to their corrective benefit; annual deterioration occurred in
  all three scored years. This did not isolate a causal news or financial effect.
- Proposed causes remained hypotheses: weak transferable signal, magnitude bias,
  short news windows relative to annual targets, sparse financial coverage and
  target tails. The proposed 209/130-column factorial comparison would separate
  news and financial effects using 48 additional validation fits; it was not
  approved or implemented by the old brief. Neither calibration nor deleting
  losing examples follows automatically from the diagnostic.
- Historical classifier AUCs concern different targets, dates and fitted bundles.
  They do not establish a currently accepted earlier strong model, net SPY
  outperformance or authorization for an alternative production API.

Retained artifact locators and original pins from that brief follow. These are
historical citations; this documentation consolidation did not replay their data.

Full target corpus. Path: data/research/investment_full_targets_20261109/corpus_0a1ca7d/final/_manifest.json.

SHA-256: 9b70c348ab17738f5874d42c51bff6af386d94061ef383385bdd6ead5bcad4c3

Joined dataset. Path: data/research/investment_dataset_20261110/joined_30b54bf/dataset/_manifest.json.

SHA-256: 9c19fbf3a9f716b3f82b9958c094f9752f477952f5df8c4cb797123e765c1280

Completed training. Path: data/research/investment_training_20261110/models_30b54bf/_manifest.json.

SHA-256: 16eda742f2b6c0b95d25c03796185aac447c6c348f6710f6b7037640675516b3

Independent post-fit audit. Path: Chat outputs/investment-models-independent-postfit-audit-20261110.json.

SHA-256: b139c5e84d98c332aec3598c4751dedb35b5d4d8f2b414630604839cd94404dd

Completed prediction-error diagnostic. Path: data/research/investment_prediction_errors_20261110/diagnostics_770d0db/_manifest.json.

SHA-256: 2e570a4b714e0305d11909fe23ee48eeb116114be42ada16331070a92ce86981

Historical attachments should include models/sp500_6m_next_day_big_up_v2_20260708_candidate.joblib.manifest.json; the corresponding next_week_big_up manifest; models/edge_rebuild_intraday_candidate_20260802_v1/evaluation.json; and data/models/edge_rebuild_intraday_candidate_v3/evaluation.json with their model cards. Relevant Git excerpts are 87b3aad for the contemporaneous baseline record, 53c51c2 for the workflow replacement, 5ebfa83 for snapshot-feature removal, 0332b16 for intraday V3 completion, and 3b2bff5/7bb128c for the later SPY-relative objective and measured swing regressions.
