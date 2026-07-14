# Agent Harness Optimization Research Strategy

## Overview

This document outlines the research design, experimental structure, and implementation plan for a study on how varying harness optimization strategies affect an agent's ability to solve complex, multi-turn problems. It is intended to serve as the authoritative reference for both the GitHub repository and Claude Code implementation sessions.

---

## Research Question

**How do varying harness optimization strategies affect an agent's ability to solve complex, multi-turn financial modeling problems — measured jointly by portfolio P&L performance and computational cost (tokens consumed, latency)?**

Secondary questions:
- Does structured long-term memory outperform naive context injection (all-prior-episodes Ralph Wiggum) on the same task?
- At what point does additional harness complexity yield diminishing returns relative to cost?
- Under an identical autonomous harness, does a sparse MoE model or a dense model make the better primary agent — jointly on economic performance, task-completion reliability, and inference throughput?

---

## Hardware & Model Stack

### Hardware
- **Machine:** ASUS Ascent GX10 (GB10 Grace Blackwell)
- **RAM:** 128GB unified memory
- **Stack:** CUDA/NVIDIA (full CUDA support, not ROCm)
- **Inference framework:** vLLM

### Models
| Role | Model | Architecture | Active Params | Notes |
|---|---|---|---|---|
| Candidate Primary (T1a) | Qwen3.6-35B-A3B | Sparse MoE | ~3B active | Fast, low inference cost |
| Candidate Primary (T1b) | Qwen3.6-27B | Dense | 27B | Stronger reasoning, higher cost |

Both models are served locally via vLLM. Memory on the shared 128GB unified-memory box only comfortably fits one model at generous context at a time (see [Open Questions & Future Work](#open-questions--future-work)), which turned out to align naturally with the study design below: T1a and T1b never need to run concurrently, only sequentially.

### Primary Model Selection (Treatments 1a/1b)
There is no fixed "primary model" assumption for T2 onward — which model the rest of the ladder runs on is an empirical decision made after T1a and T1b each complete N=20 valid holdout episodes under the identical `create_deep_agent()` harness. The decision rule, pre-registered before either treatment's results are inspected for this purpose:

1. **Primary criterion — portfolio P&L on final holdout**, with a bootstrapped confidence interval (episode-level P&L resampled) rather than a bare point estimate, given N=20 is not large. The model with the higher mean P&L wins *if* the intervals are meaningfully separated.
2. **Tiebreaker — completion rate**, also with a bootstrapped interval, used only if the P&L intervals overlap substantially. More reliable completions compound into more usable data for every downstream treatment, so reliability is treated as the tiebreak, not a co-equal primary axis.
3. **Secondary consideration — inference throughput.** Decode-phase tokens/sec is read directly from vLLM's own Prometheus metrics (`/metrics`, e.g. `vllm:inter_token_latency_seconds`), snapshotted before/after each treatment's run — this isolates pure generation speed from tool-execution and sandbox time, which dominate episode wall-clock and would otherwise confound any throughput comparison derived from episode-level `duration_s`. If P&L and completion rate come out close, materially faster decode throughput on the sparse MoE model is a legitimate reason to prefer it, since it directly changes how much of the paper's compute budget the remaining treatments consume.
4. If the outcome is genuinely ambiguous even after all three axes (a real possibility at N=20), that ambiguity is reported as a finding in its own right rather than resolved by fiat.

This structurally replaces the advisor design considered earlier (see Future Work) — instead of running both models jointly with the dense model as an escalation path, the ladder now commits to a single winning primary model for T2 onward, decided empirically rather than assumed.

---

## Task Definition

### The Problem
The agent is tasked with designing a Python-based credit risk assessment and pricing optimization system. Specifically:

1. **Risk Model:** Build a survival-style model predicting probability of default and time-to-default given applicant features and loan terms
2. **Pricing Function:** Given a risk estimate and a customer sensitivity model, determine optimal offer terms (amount, rate) to maximize expected portfolio P&L
3. **Evaluation:** Apply the risk model and pricing function to a holdout population, simulate customer acceptance via the sensitivity model, and compute realized P&L based on actual historical outcomes

### Why This Task
- Objectively measurable outcome (P&L) that grounds evaluation in economic reality
- Requires multi-turn, multi-step reasoning across data exploration, feature engineering, modeling, and optimization
- Rich enough that long-term memory about what approaches work has genuine value
- Directly relevant to real-world fintech applications (revenue-based financing, consumer lending)
- Assumption explicitly noted: offer pricing/terms do **not** causally affect individual borrower default probabilities — each borrower's `event` (default/survive) is fixed from historical LendingClub outcomes and does not change based on the rate we offer them. We do not model payment-stress effects (where high rates cause otherwise-safe borrowers to default).
- However, offered rate **does** affect portfolio-level default rates through **adverse selection**: the two sensitivity models cause above-market rates to disproportionately attract credit-constrained borrowers (those who would default), causing non-defaulters with outside options to walk away. This is a selection composition effect, not a causal one — the distinction is meaningful and noted in the paper's limitations.

---

## Dataset

### LendingClub Loan Data
- **Source:** Kaggle (publicly available, originally released by LendingClub)
- **Coverage:** Apr 2008–Sep 2018 (single compiled file, not annual cohorts)
- **Size:** 2,260,701 loan records; 104 pre-origination features retained after leakage/PII removal

### Key Fields
| Field | Role |
|---|---|
| `loan_amnt`, `term`, `int_rate`, `installment` | Offer terms |
| `fico_range_low`, `fico_range_high` | Credit quality signal |
| `dti`, `annual_inc`, `emp_length` | Borrower capacity |
| `delinq_2yrs`, `inq_last_6mths`, `pub_rec` | Derogatory history |
| `revol_bal`, `revol_util`, `open_acc` | Credit utilization |
| `home_ownership`, `verification_status`, `purpose` | Categorical risk signals |
| `loan_status` | Event indicator (Charged Off = default event) |
| `issue_d`, `last_pymnt_d` | Used to compute observed survival time in months |
| `grade`, `sub_grade` | LendingClub's internal risk grade |

### Survival Target Construction
From `loan_status`, `issue_d`, and `last_pymnt_d`:
- **Event (default=1):** `loan_status == "Charged Off"`
- **Censored (default=0):** `loan_status == "Fully Paid"` or `"Current"`
- **Observed time:** months between `issue_d` and `last_pymnt_d`

This yields a proper right-censored survival dataset suitable for Cox PH, discrete-time hazard models, or gradient boosted survival models.

### Three-Way Temporal Data Split
| Partition | Vintage | Purpose |
|---|---|---|
| Training | 2007–2014 | Agent model fitting, GEPA optimization |
| Validation OOT | 2015–2016 | Within-loop performance feedback (episode index scores) |
| Final Holdout OOT | 2017–2018 | Terminal evaluation only — never seen during any loop iteration |

**Critical:** The final holdout is not touched until the last evaluation of each agent variant. This prevents adaptive overfitting — particularly important for the long-term memory variant, which could otherwise learn the specific characteristics of the OOT population across hundreds of loop iterations (a form of indirect leakage).

**Observed split sizes and event rates:**
| Partition | Rows | Event Rate | Note |
|---|---|---|---|
| Training | 466,344 | 16.6% | Fully matured loans |
| Validation OOT | 855,491 | 16.8% | Fully matured loans |
| Final Holdout | 938,793 | 5.1% | Lower rate due to loan maturity — 2017–2018 cohort had insufficient time to fully charge off at data snapshot. Correct for evaluation; survival model trained on mature loans. |

---

## Customer Sensitivity Model

A synthetic parametric acceptance model made available to the agent as a callable tool. Given an applicant's features and a proposed offer, it returns the probability that the customer accepts.

**Why synthetic, not data-trained:** The public LendingClub dataset contains only funded loans — applications that both LendingClub approved and the borrower accepted. There is no "LendingClub made an offer, borrower declined" signal in the data. Training a sensitivity model on funded loans would conflate LendingClub's underwriting decisions with borrower acceptance behavior. A synthetic model with calibrated parameters avoids this selection bias, provides a known ground truth that can be reasoned about analytically, and is fully reproducible.

### Two-model adverse selection design

The simulation ships **two** sensitivity models with different rate elasticities. The design reflects a key empirical reality in consumer credit: when a lender charges above-market rates, the borrowers who still accept are disproportionately those who cannot obtain credit elsewhere — i.e., higher actual credit risk within the same grade tier.

**Model formulation (shared structure, different β_spread):**
```
P(accept | applicant, offer, type) = sigmoid(
    α₀
    - β_spread(type) * max(0, offered_rate - market_rate_for_grade)
    - β_burden       * (loan_amnt / annual_inc)
    + β_match        * (loan_amnt / funded_amnt)
    + ε
)
```

| Model | File | β_spread | Interpretation |
|---|---|---|---|
| Non-defaulter | `sensitivity_model_nondefaulter.pkl` | 16.0 | Has outside credit options; walks away from above-market offers |
| Defaulter | `sensitivity_model_defaulter.pkl` | 8.0 | Credit-constrained; accepts even costly offers |

This is evaluated as a **backtest** on historical loans, so each borrower's realized `event`
outcome is known ground truth. The acceptance probability and expected loss both key off
`event` directly — not off the agent's own risk model estimate `p_default_hat`:

```
p_accept_eff  = event × p_accept_bad + (1 − event) × p_accept_good
expected_loss = event × p_accept_bad × balance_at_t
```

`p_default_hat` is deliberately excluded from this formula. Early drafts blended acceptance
by `p_default_hat` instead of `event`; that let a risk model biased toward predicting
universally high default probability inflate `p_accept_eff` (via `p_accept_bad`, the more
lenient curve) across the *entire* population without a matching rise in `expected_loss`
(which only depends on `event`) — reopening the flat-high-rate degenerate strategy the
two-model design exists to close, this time via a miscalibrated risk model instead of a
flat rate. `p_default_hat` still does real work: it drives the agent's pricing function
(`offered_rate` per applicant) and loan ranking/selection under the capital cap. It just
never gates the realized acceptance/loss mechanics, since those are already determined by
history.

This creates genuine adverse selection: charging above-market rates selectively attracts the worst borrowers within each grade. **Flat-rate strategies (e.g., 36% to all Grade C–F) are therefore suboptimal** — the effective default rate in the accepted pool rises sharply with rate spread.

**Adverse selection calibration (β_spread_bad=8, β_spread_good=16):**
| Grade | Base default rate | Eff. default rate at 36% | Delta |
|---|---|---|---|
| C | 18.0% | ~40% | +22 pp |
| D | 23.9% | ~39% | +15 pp |
| E | 30.6% | ~37% | +6 pp |
| F | 35.2% | ~37% | +2 pp |

Grade C and D are most penalised because their market rates (19%, 24%) are furthest below the 36% ceiling, maximising adverse selection. Grade F is near-neutral because its market rate (34%) is close to 36%.

**Shared parameters (see `data_pipeline/sensitivity_model.py`):**
- `COST_OF_CAPITAL = 0.16`, `SERVICING_MARGIN = 0.03`, `EXPECTED_LOSS_BUFFER = 0.02` → `MIN_VIABLE_RATE = 0.21`
- `ALPHA_0 = 0.20`, `BETA_BURDEN = 1.5`, `BETA_MATCH = 0.50`, `NOISE_STD = 0.30`

**Why the within-grade adverse selection signal is absent from the raw LendingClub data:** Analysis of the training set showed defaulters and non-defaulters within the same sub-grade received nearly identical rates (spread < 0.12 pp, vs ~1 pp within-sub-grade std dev). LendingClub's granular 35-sub-grade system pre-absorbed the between-borrower rate variation, leaving no empirical calibration signal for differential acceptance elasticity. The two-model parameters are therefore theoretically grounded (Karlan & Zinman 2008) rather than data-estimated.

**Fairness note:** `zip_code` is retained as a feature. It is predictive (local economic conditions correlate with default risk) but carries disparate impact risk due to historical redlining patterns. If the agent relies heavily on this feature, it warrants discussion in the paper's limitations section.

**Cost of capital / risk-based required rate:** The simulation explicitly models a non-bank fintech lender with a high cost of capital (16% blended warehouse + ABS funding) plus a 3% servicing margin. Rather than a flat floor, each applicant has a borrower-specific **required rate** — the minimum rate needed to break even given their predicted default risk:

```
risk_margin    = (p_default_hat × LGD) / ((1 − p_default_hat) × AVG_TERM_YEARS)
required_rate  = COST_OF_CAPITAL + SERVICING_MARGIN + risk_margin
```

`AVG_TERM_YEARS = 3.85` (empirical weighted average loan term). `LGD` (loss-given-default — the average fraction of original principal still outstanding when a loan of this grade charges off) is calibrated per grade from `train.parquet`, using the same amortization math as the P&L formula, applied to actual historical defaulters at their real historical rate:

| Grade | LGD | Grade | LGD |
|---|---|---|---|
| A | 0.51 | E | 0.70 |
| B | 0.55 | F | 0.74 |
| C | 0.60 | G | 0.77 |
| D | 0.65 | | |

Riskier grades default earlier relative to their term on average, leaving more principal outstanding — hence higher LGD. If `required_rate > 0.36` (the regulatory ceiling), the applicant is **declined** — no rate exists that makes them profitable within the legal maximum. This is the "kicker": for a real chunk of the worst-scored individuals in Grade F/G, the risk math genuinely doesn't work, independent of any pricing cleverness.

At grade-average default rates, required rates come out to: A≈19.8%, B≈20.8%, C≈22.4%, D≈24.3%, E≈27.0%, F≈29.4%, G≈33.4% — all technically viable, but Grade A/B's required rate sits far above their market rate (8–14%), so the sensitivity models' elasticity does the work of excluding them (borrowers walk) rather than the decline logic. Decline logic mainly bites the worst-scored tail of Grade F/G, where individual `p_default_hat` pushes past the grade average enough to cross the 36% ceiling.

| LendingClub Grade | Approx Market Rate | Competitive for This Lender? |
|---|---|---|
| A (750+ FICO) | 8–11% | No — required rate (~20%) far exceeds market; borrowers walk |
| B (700–749) | 12–16% | No — required rate (~21%) still well above market |
| C (670–699) | 17–21% | Marginal — required rate (~22%) close to market, thin margin |
| D (640–669) | 22–26% | Yes — required rate (~24%) near market, healthy margin available |
| E (600–639) | 27–32% | Yes — sweet spot, required rate (~27%) comfortably under market |
| F/G (<600) | 33%+ | Yes for most; worst-scored tail within grade gets declined outright (required rate > 36%) |

**Design notes:**
- Moderate stochasticity baked in — enough that the agent must reason in expected value terms, not enough to wash out the learning signal
- Both models held fixed across all experimental variants (not re-trained between episodes)
- Both models callable via `sensitivity_model_query` tool or loaded directly as pickles

**Portfolio constraint:** The agent operates under a **capital cap only, no volume floor**. Earlier designs paired a small $15M cap with a 400-loan floor, sized around an "~1,000 applicants per episode" framing that was never actually enforced in code — agents evaluated against the full ~855K-row val set regardless. Against a pool that size, a $15M cap (~750-950 loans) let greedy ratio-based selection cherry-pick an extreme favorable tail (near-zero-acceptance "free options," and true defaulters who happened to default near loan maturity, leaving almost no outstanding balance) regardless of pricing strategy — a mechanism invisible in aggregate P&L until forced into by real volume. A large cap sized to the full pool ($2.5B — see below) removes the ability to dodge the tradeoff via cherry-picking; volume is left uncapped from below because it's a symptom of good risk-based pricing, not a target in itself — a real lender doesn't chase a headcount, it chases return on deployed capital.

**Decision vs. evaluation phases — avoiding lookahead bias:** Which loans get offered and which get funded under the cap must be decided using only information available ex-ante (applicant features, the agent's own `p_default_hat`) — never the realized `event`/`observed_time` outcome. Ranking loans by their *realized* P&L to decide who to fund lets the selection cherry-pick applicants because their future is already known (e.g., preferentially "funding" borrowers who happen to not default, or who default very late in their term) — information no real underwriter has at decision time. Realized `event`/`observed_time` is only used afterward, to evaluate the P&L of a portfolio already selected — legitimate, since that's what a backtest is for.

**Baseline parameters (documented in `agent/app/prompts/system.py`; `configs/base.yaml`'s `portfolio` section mirrors these but is not read by any code path):**
- Capital cap: **$2,500,000,000** per episode
- No volume floor
- Applicant pool: full `val.parquet` (no subsampling — ~855K rows, ~459K in the viable Grade C–G segment)

---

## Evaluation Metrics

### Primary
- **Simulated Portfolio P&L** on validation OOT (within-loop feedback) and final holdout (reported results)
- Computed as: sum of (interest income − principal loss) across accepted loans, weighted by effective acceptance probability from the two sensitivity models. Uses proper loan amortisation: loss is the **remaining outstanding balance at default** (not original principal), and interest income reflects **actual amortised payments received** (not simple rate × time). A month-50 default on a 60-month loan loses ~30% of original principal, not 100%.

### Secondary
- **Concordance Index (C-statistic)** on survival model
- **Acceptance Rate** — volume of offers taken
- **Total tokens consumed** per episode and per variant to plateau
- **Wall-clock latency** per episode
- **Iterations to plateau** in Ralph Wiggum loop
- **Decode throughput (tokens/sec)** — read from vLLM's server-side Prometheus metrics, not derived from episode wall-clock (which is dominated by tool-execution/sandbox time, not model generation). Primarily used for the T1a/T1b model-selection decision; also relevant to characterizing overall study compute cost.

### Statistical Treatment
Given per-treatment N is modest (20 for the non-loop treatments), point-estimate comparisons between treatments (P&L, completion rate) are supplemented with bootstrapped confidence intervals (episode-level resampling) rather than reported as bare means. This applies to the T1a/T1b model-selection decision and, where sample size allows, to the GEPA promotion decisions in Treatment 2 (see [GEPA Optimization](#gepa-optimization)). A real possibility at this N is that some comparisons will not reach clean statistical separation — that is itself a reportable finding, not a result to be papered over by picking a favorable point estimate.

### Reported Results Structure
Every variant is evaluated on the **final holdout only**. Within-loop validation scores are logged for analysis but not reported as primary results. This ensures all variants are compared on equal footing regardless of loop iteration count.

---

## Experimental Design

### The Ralph Wiggum Loop
The universal evaluation harness applied to every agent variant after the baseline. In each iteration:
1. Agent reviews its episode index (prior performance + skill locations)
2. Agent builds or refines its risk model and pricing function in the sandbox
3. Agent's function is evaluated on the validation OOT sample
4. Results are logged to the episode index
5. Repeat until performance plateaus

The loop runs until plateau is detected (e.g., less than X% P&L improvement over 3 consecutive iterations).

**Two flavors of the loop are tested:**
- **Single-prior:** Agent sees only the immediately preceding episode's result and code location. Risk: local optima traps — agent may get stuck on a suboptimal solution with no visibility into earlier, potentially better approaches.
- **All-prior:** Agent sees the full episode index across all iterations. Enables pattern synthesis across the full performance history. This is a manually-injected analog of long-term memory — a key comparison point against the structured memory subagent.

### Episode Index Format (All-Prior Variant)
Each entry in the episode index passed to the agent in context:

```
Episode N | P&L: $Xk | C-stat: X.XX | Acceptance Rate: XX% | 
Approach: "<one sentence summary of strategy>" | 
Hypothesis: "<what the agent intended to improve>" |
Skill: /skills/episode_N_pricing.py
```

This keeps the index compact (2-3 lines per episode) so even 20+ episodes remain manageable in context. The agent retrieves actual code from the filesystem only when it chooses to build on a specific prior attempt.

### GEPA Optimization
GEPA runs immediately after the baselines and the T1a/T1b model-selection decision (Treatment 2), before any loop treatments, targeting whichever model won the bake-off:
- GEPA runs offline on the **training partition only**
- Produces a locked optimized prompt set used for all subsequent treatments (T3–T6)
- GEPA is a one-time calibration step — prompts are fixed before loop evaluation begins, eliminating the confound of simultaneous prompt optimization and performance measurement
- Positioning GEPA before the loops ensures that loop gains (T3, T4) cannot be dismissed as "compensating for a suboptimal prompt" — all loop results are on optimized prompts

**Pareto multi-objective framing, not a blended scalar.** Candidate prompts are compared on three independent axes rather than collapsed into one optimization target:
1. **Performance** — portfolio P&L on holdout
2. **Reliability** — episode completion rate (ratio of valid, scored episodes to attempts)
3. **Efficiency** — *pathological* tool-call error rate. This deliberately excludes ordinary try-fail-fix debugging cycles (edit_file string-not-found, a syntax error caught and corrected, a retrained model after a first attempt underperforms) — that's healthy, expected agent behavior, observed throughout the T1a baseline run. The metric instead targets identifiable failure patterns: runs of consecutive identical tool calls (the same command, same error, no variation — the signature of a genuinely stuck agent, as seen in one T1a episode that required manual intervention), not raw error counts.

A candidate survives if it is **not dominated** by any existing prompt on the frontier — i.e. no existing candidate beats it on all three axes simultaneously. This is a purely arithmetic decision once trial metrics are in; it is not a judgment call.

**Two distinct roles for the LLM-as-judge pattern, not one:**
- **Candidate generation** (a genuine judgment call): either an *informed mutation* — analyze a sample of an existing prompt's episode traces, diagnose a specific weakness, propose a targeted edit — or an *evolution* — combine two frontier prompts that are strong on different axes, attempting to preserve both strengths in one merged candidate. Claude performs this role.
- **Promotion decision** (not a judgment call): whether a completed trial's measured metrics place it on the non-dominated frontier. Fully objective and reproducible from the numbers alone.

**Two-stage trial structure**, both stages scored against holdout (dataset choice and rollout count are orthogonal — using holdout for both means the only difference between stages is sample size, not what's being measured):
1. **Screening** — a small number of rollouts (starting point 3-5, to be sized empirically once T1a's full-run variance is characterized — see [Statistical Treatment](#statistical-treatment)). A deliberately cheap, leaky filter: false negatives (discarding a good mutation too early) are the acceptable failure mode; false positives (promoting a mediocre mutation to the expensive confirmation stage) are the costly one.
2. **Confirmation** — a larger rollout count (starting point ~20) for any candidate that clears screening, producing the actual Pareto-comparison numbers.

---

## Treatment Ladder (Ablation Study)

Each treatment is a cumulative addition to the prior level. Treatments 3–6 all run on GEPA-optimized prompts (locked after Treatment 2), on whichever model T1a/T1b selects (see [Primary Model Selection](#primary-model-selection-treatments-1a1b)). No treatment in this ladder uses the advisor pattern considered earlier — see [Open Questions & Future Work](#open-questions--future-work) for why it was dropped in favor of the T1a/T1b bake-off.

| # | Variant | Framework | Model | Loop | GEPA | RLM STM | LT Memory |
|---|---|---|---|---|---|---|---|
| 0 | Vanilla ReAct | `create_agent()` | 35B-A3B | — | — | — | — |
| 1a | DeepAgent (batteries on) | `create_deep_agent()` | 35B-A3B | — | — | — | — |
| 1b | DeepAgent (batteries on) | `create_deep_agent()` | 27B | — | — | — | — |
| 2 | GEPA-optimized DeepAgent | `create_deep_agent()` | [T1a/T1b winner] | — | ✓ | — | — |
| 3 | GEPA + Loop (single-prior) | `create_deep_agent()` | [T1a/T1b winner] | Single | ✓ | — | — |
| 4 | GEPA + Loop (all-prior) | `create_deep_agent()` | [T1a/T1b winner] | All | ✓ | — | — |
| 5 | GEPA + RLM + Loop | `create_deep_agent()` | [T1a/T1b winner] | All | ✓ | ✓ | — |
| 6 | Full System (all components) | `create_deep_agent()` | [T1a/T1b winner] | All | ✓ | ✓ | ✓ |

**T0, T1a, T1b, and T2** are each run N=20 independent episodes (no loop) to establish performance distributions. T1a/T1b jointly decide the primary model for T2–T6 (see decision rule above); whichever wins becomes the anchor for statistical comparisons from T2 onward.

**Loop mode for T5–T6:** T3 vs T4 establishes which loop mode wins. T5–T6 use the winner (expected: all-prior). If T3 surprises, a T5a single-prior variant can be run as a follow-on.

**Key pairwise comparisons the ladder enables:**
| Comparison | What it isolates |
|---|---|
| T0 vs T1a | Full DeepAgent battery value (todos, filesystem, summarization, subagents) vs vanilla ReAct |
| T1a vs T1b | Model selection: sparse MoE vs dense, identical harness — decides the primary model for T2–T6 |
| T1a/T1b vs T2 | Pure GEPA prompt optimization contribution, on the winning model |
| T2 vs T3 | Pure single-prior loop contribution, on optimized prompts |
| T3 vs T4 | All-prior vs single-prior context injection |
| T4 vs T5 | RLM short-term memory contribution |
| T5 vs T6 | Structured long-term memory vs naive all-prior injection |

---

## Agent Architecture

### Framework
- **T0 orchestration:** LangChain `create_agent()` (vanilla ReAct, v1.x) — no batteries, fully custom tool set
- **T1–T6 orchestration:** LangChain `create_deep_agent()` (deepagents v0.6+) — batteries included, custom supplementary tools layered on top
- **Inference:** vLLM on GX10, local OpenAI-compatible endpoint
- **Observability:** Langfuse (MIT licensed, self-hostable; first-class LangGraph integration for trace logging, cost tracking, and dataset evaluation)
- **Sandbox:** Controlled Python execution environment with filesystem access

### Agent Tools
| Tool | Present in | Description |
|---|---|---|
| `code_executor` | T0–T6 | Python sandbox with project PYTHONPATH; custom-built for our data stack |
| `filesystem_read/write` | T0 only | Custom tools with explicit path controls (replaced by deepagents built-ins in T1+) |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | T1–T6 | deepagents built-in filesystem tools (scoped to skill_dir + data/ via FilesystemPermission) |
| `execute` | T1–T6 | deepagents built-in shell execution |
| `write_todos` | T1–T6 | deepagents built-in todo list |
| `task` | T1–T6 | deepagents built-in subagent spawning |
| `sensitivity_model` | T0–T6 | Call customer acceptance model with applicant + offer features |
| `query_long_term_memory` | T6 only | Subagent interface to Mem0 structured learning store |

### Short-Term Memory (RLM Subagent, Treatments 5-6)
A Recursive Language Model subagent with access to the full running context. Paired with aggressive compaction middleware to prevent context window bloat while preserving coherence across long episodes.

### Long-Term Memory Subagent (Treatment 6)
- **Framework:** Mem0 (open source, self-hostable), running against the local vLLM endpoint for embeddings and extraction
- **Reflection Agent:** Post-episode LangGraph node that extracts structured learning objects from the episode trace and writes to Mem0 via `client.add()`
- **Retrieval:** `query_long_term_memory` tool wraps `client.search()` — primary agent sends a natural language query, Mem0 returns semantically relevant learnings ranked by relevance
- **Handled by Mem0 internally:** Embedding generation, deduplication, temporal relevance decay, and learning clustering
- **Human review queue:** Optional gate before learnings are promoted to the long-term store (configurable)

---

## Implementation Plan

### Phase 1 — Infrastructure
- [x] Set up vLLM serving both models on GX10
- [x] Build agent scaffold: `create_agent()` (T0) and `create_deep_agent()` (T1–T6) with tool definitions
- [x] Implement controlled Python sandbox with filesystem (`code_executor` tool)
- [x] Build and calibrate synthetic customer sensitivity model (parameterize CoC floor, market rate tiers, β coefficients)
- [x] Construct survival target variables from LendingClub dataset
- [x] Implement three-way temporal data split
- [x] Build episode index logger and skill artifact storage (Ralph Wiggum loop + JSONL index)

### Phase 2 — Baselines
- [ ] Run Treatment 0 (vanilla create_agent, N=20 runs), record distribution
- [ ] Run Treatment 1a (deepagent 35B-A3B, N=20 runs), record distribution
- [ ] Run Treatment 1b (deepagent 27B, N=20 runs), record distribution

### Phase 3 — Model Selection & GEPA Optimization
- [ ] Decide primary model from T1a/T1b per the pre-registered decision rule (P&L with bootstrapped CI, completion-rate tiebreaker, throughput as secondary consideration)
- [ ] Integrate GEPA library or implement custom GEPA loop (Pareto frontier: P&L, completion rate, pathological tool-error rate; Claude as mutation/evolution candidate generator; objective non-dominance promotion)
- [ ] Size the screening-stage rollout count empirically from T1a's observed episode-to-episode variance
- [ ] Run GEPA optimization on training partition, targeting the selected model
- [ ] Lock optimized prompts
- [ ] Run Treatment 2 (GEPA-optimized baseline, no loop)

### Phase 4 — Loop Treatments
- [ ] Implement Ralph Wiggum loop (single-prior flavor)
- [ ] Run Treatment 3
- [ ] Implement all-prior episode index injection
- [ ] Run Treatment 4

### Phase 5 — Memory
- [ ] Implement RLM subagent + compaction middleware
- [ ] Run Treatment 5
- [ ] Implement long-term memory subagent (Mem0 + post-episode reflection node)
- [ ] Run Treatment 6

> **Scope note:** Treatments 1a, 1b, 2–4 are the load-bearing contributions for publication. Treatments 5–6 strengthen the paper but could be deferred to follow-on work if implementation scope becomes prohibitive. Decide at the end of Phase 4.

### Phase 6 — Evaluation & Paper
- [ ] Run all variants on final holdout OOT
- [ ] Compute P&L, C-statistic, token cost, latency per variant
- [ ] Generate P&L curves across loop iterations per variant
- [ ] Generate cost-efficiency Pareto frontier (final P&L vs. total token cost)
- [ ] Statistical significance testing vs. baseline distribution
- [ ] Write paper

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| T0: `create_agent()` vanilla baseline | Zero-effort ceiling — shows what deepagents batteries add before any custom ablation components enter. Publishable at LangChain-adjacent venues as a framework evaluation. |
| T1–T6: `create_deep_agent()` | Batteries-included baseline; our ablation components layer on top. FilesystemPermission scopes built-in tools to skill_dir and data/. |
| T5–T6 use all-prior loop only | T3 vs T4 establishes which loop mode wins; subsequent treatments use the winner to keep the comparison clean. T5a (single-prior + RLM) can be run as a follow-on if interaction effects are of interest. |
| Three-way data split | Prevents adaptive overfitting of long-term memory to OOT population |
| Ralph Wiggum as universal harness | Enables cost/iteration comparison across all variants on equal footing |
| GEPA before loops (T2), not during | Eliminates confound between prompt optimization and loop performance measurement; all loop comparisons are on optimized prompts, preventing dismissal of loop gains as "compensating for bad prompts" |
| Dual baseline (T1a: 35B-A3B, T1b: 27B) decides the primary model empirically | Rather than assuming the sparse MoE model going in, T1a/T1b run identically and the winner (by the pre-registered P&L/completion-rate/throughput rule) carries forward through T2–T6. Replaces an earlier design where 35B-A3B was assumed as primary by construction. |
| GEPA as Pareto frontier, not a blended scalar | P&L, completion rate, and pathological tool-error rate are kept as separate axes rather than combined into one score — a blended scalar hides which dimension actually improved or regressed, and different stakeholders (business performance vs. research reliability) care about different axes |
| Claude generates GEPA candidates; promotion is arithmetic | Separates the one genuinely judgment-based step (proposing a mutation or evolution) from the survive/die decision (objective non-dominance on measured metrics) — keeps the actual selection criterion reproducible and not dependent on judge consistency across generations |
| Tool-call error rate targets pathological loops, not raw error count | Ordinary try-fail-fix debugging (a syntax error caught and fixed, a retrained model after underperforming) is healthy agent behavior observed throughout T1a — penalizing raw error counts would select against careful iteration. The metric instead targets identifiable failure signatures like runs of consecutive identical tool calls. |
| Decode throughput read from vLLM server metrics, not episode wall-clock | Episode `duration_s` is dominated by tool-execution/sandbox time (model training, pandas operations), not model generation — dividing tokens by episode duration would badly understate and confound real model speed. vLLM's own Prometheus histograms (`inter_token_latency_seconds`, etc.) isolate pure decode-phase throughput with no additional harness instrumentation. |
| P&L as primary metric | Grounds results in economic reality rather than model accuracy proxies |
| Capital cap sized to full applicant pool, no volume floor | A cap sized proportionally to the ~855K-row val set (vs. the pool) prevents ratio-based selection from cherry-picking an unrealistic favorable tail; volume is left as an outcome of risk-based pricing rather than a forced target |
| Single-prior vs. all-prior loop | Tests local optima trapping; sharpens contrast between naive context injection and structured memory |
| Synthetic sensitivity model | LendingClub contains only funded loans — no borrower reject signal. Synthetic model avoids selection bias and provides known ground truth for analytical verification |
| Cost of capital floor | Non-bank fintech cannot competitively serve prime borrowers; CoC floor creates realistic market segmentation the agent must discover. Maps to real-world non-bank lending economics |
| Mem0 for LTM storage | Drop-in LangGraph-compatible memory layer replaces custom DynamoDB + reflection pipeline; handles embedding, deduplication, and retrieval internally |
| Langfuse for observability | MIT-licensed, self-hostable LLM observability with LangGraph integration. LangSmith requires Enterprise license for self-hosting |

---

## Open Questions & Future Work

- **Dense-advisor pattern (Anthropic Advisor Strategy) considered and deferred.** An earlier design ran the sparse MoE model as primary with the dense 27B model as a selectively-invoked advisor (Treatment 5 in an earlier version of this ladder). Dropped for a concrete, empirically-discovered reason rather than a scoping guess: the shared 128GB unified-memory box only comfortably serves one model with a generous context window at a time — T1a alone needed `vllm-primary`'s context roughly doubled (65,536 → 131,072 tokens) to run its `create_deep_agent()` harness reliably, which required freeing the entire memory budget the advisor model would otherwise occupy. Running both simultaneously at reduced context each was considered (e.g. `--kv-cache-dtype fp8` to roughly halve KV-cache memory) but not pursued once the T1a/T1b model-selection restructure made it unnecessary — the ladder no longer needs both models loaded together at all. If a future extension revisits the advisor pattern, fp8 KV cache is the natural first thing to try before assuming the hardware can't support it.
- Capital cap set to $2.5B (no volume floor) as baseline — this was tuned empirically (T0 v1/v2 runs showed a $15M cap against the full ~855K-row val set allowed degenerate cherry-picking regardless of pricing strategy); may need further tuning after Treatment 1 baseline runs reveal agent behavior
- Whether GEPA should optimize per-subagent prompts independently or the full compound system
- Whether the reflection agent should run post-episode (cleaner) or on a timer (more autonomous)
- Extension to multi-agent competitive setting (multiple pricing agents competing for same borrower pool)
- Generalization test: does the full system transfer to a different credit domain (e.g., revenue-based financing for SMBs) without re-running GEPA?
- Statistical significance approach and baseline run count N — depends on observed variance in Treatment 1 distribution
- **The harness likely understates any advantage of survival modeling over plain classification.** The task frames itself around survival targets (`event`/`observed_time`) and offers survival-specific tooling (pycox, lifelines), but the only thing the pricing economics ever consume is a single scalar — `p_default_hat` (and, via `required_rate`, a flat population-average assumed loan duration). Harrell's C-statistic (the required eval metric) only rewards rank ordering, which a plain classifier can match without handling censoring correctly at all. A survival model's structural advantage — a full hazard curve, not just an endpoint probability — has essentially nowhere to pay off in the current design, since the decision layer never asks a duration-conditional question. Considered fixing this by charging `cost_of_capital` as a real dollar cost proportional to `avg_balance × duration_deployed` (currently it only gates the `required_rate` pricing floor and is never actually subtracted from realized P&L) — this would make forecasting *when* a loan is likely to exit, not just *whether* it defaults, economically load-bearing. A quick validation swept required-rate pricing strategies with this charge added: P&L could swing from strongly positive to roughly breakeven/negative depending on the assumed duration, a large enough effect to change which strategy is optimal (not just rescale results) — while investigating, found `observed_time` in `train.parquet`/`val.parquet` is heavily right-censored (only ~26% of "non-default" loans are observed within 95-100% of their contractual term; the rest are still-current-as-of-snapshot, not genuinely resolved), which biases any naive duration estimator and calls the validation's exact magnitude into question. Doing this correctly needs a proper censored-duration estimator (e.g. Kaplan-Meier restricted mean survival time), not naive averaging — real additional work, not a quick fix. Deferred rather than implemented: T0-T6 all run under the same (duration-blind) economics, so cross-treatment comparisons remain valid, but the study is not well-positioned to detect whether agents that reach for survival methods are meaningfully rewarded for it. Flagged as future work rather than a limitation of the current results, since every treatment faces the identical gap.

---

## References

- GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (arXiv 2507.19457, ICLR 2026 Oral) — UC Berkeley, Stanford, MIT, Databricks
- Anthropic Advisor Strategy (April 2026)
- LendingClub Loan Dataset (Kaggle)
- Qwen3.6 Model Release (April 2026)
- MemoryAgentBench (ICLR 2026)
- AMA-Bench (ICLR 2026 Memory Workshop)
- Mem0 — Open source long-term memory layer for AI agents (github.com/mem-labs/mem0)
- Langfuse — Open source LLM observability and evaluation (langfuse.com; MIT licensed)
- Karlan, D. & Zinman, J. (2008). Elasticities of Demand for Consumer Credit. Yale Economic Growth Center Discussion Paper No. 926. — Basis for sensitivity model calibration; establishes that demand is inelastic at market rates but highly elastic above-market.
