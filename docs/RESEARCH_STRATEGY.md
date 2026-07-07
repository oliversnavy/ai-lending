# Agent Harness Optimization Research Strategy

## Overview

This document outlines the research design, experimental structure, and implementation plan for a study on how varying harness optimization strategies affect an agent's ability to solve complex, multi-turn problems. It is intended to serve as the authoritative reference for both the GitHub repository and Claude Code implementation sessions.

---

## Research Question

**How do varying harness optimization strategies affect an agent's ability to solve complex, multi-turn financial modeling problems — measured jointly by portfolio P&L performance and computational cost (tokens consumed, latency)?**

Secondary questions:
- Does structured long-term memory outperform naive context injection (all-prior-episodes Ralph Wiggum) on the same task?
- At what point does additional harness complexity yield diminishing returns relative to cost?
- Does a sparse MoE primary model paired with a dense advisor outperform a standalone dense model baseline?

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
| Primary Agent | Qwen3.6-35B-A3B | Sparse MoE | ~3B active | Fast, low inference cost |
| Advisor | Qwen3.6-27B | Dense | 27B | Stronger reasoning, higher cost |

### Dual-Model Inference Pattern
The primary agent runs the hot path. The advisor is invoked selectively — only when the primary agent determines it has hit a reasoning wall or needs a second opinion on a high-stakes decision. The advisor does not call tools and does not produce final output; it returns a plan, correction, or stop signal to the primary. This follows the Anthropic Advisor Strategy pattern (April 2026).

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
- Assumption explicitly noted: offer pricing/terms are not modeled as causal to default conditions — observed LendingClub outcomes are treated as fixed. This is a portfolio selection simulation, not a causal pricing model.

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

A synthetic parametric acceptance model made available to the agent as a callable tool. Given an applicant's features and a proposed offer (amount, rate, term), it returns a probability that the customer accepts the offer.

**Why synthetic, not data-trained:** The public LendingClub dataset contains only funded loans — applications that both LendingClub approved and the borrower accepted. There is no "LendingClub made an offer, borrower declined" signal in the data. Training a sensitivity model on funded loans would conflate LendingClub's underwriting decisions with borrower acceptance behavior. A synthetic model with calibrated parameters avoids this selection bias, provides a known ground truth that can be reasoned about analytically, and is fully reproducible.

**Model formulation:**
```
P(accept | applicant, offer) = sigmoid(
    α₀
    - β_rate   * (offered_rate - market_rate_for_grade)
    - β_amount * (offered_amount / annual_inc)
    + β_match  * (offered_amount / requested_amount)
    + ε
)
```
β coefficients calibrated to produce plausible acceptance behavior (30–60% average acceptance, meaningfully sensitive to rate). Calibration is informed by Karlan & Zinman (2008), who find consumer loan demand is relatively inelastic at market rates but becomes highly elastic above-market — consistent with our logistic formulation where `BETA_SPREAD` creates a sharp acceptance cliff at above-market pricing.

**Implemented parameters (see `data_pipeline/sensitivity_model.py`):**
- `COST_OF_CAPITAL = 0.16`, `SERVICING_MARGIN = 0.03`, `EXPECTED_LOSS_BUFFER = 0.02` → `MIN_VIABLE_RATE = 0.21`
- `ALPHA_0 = 0.20`, `BETA_SPREAD = 12.0`, `BETA_BURDEN = 1.5`, `BETA_MATCH = 0.50`, `NOISE_STD = 0.30`
- Serialized to `data/processed/sensitivity_model.pkl`

**Calibrated acceptance rates:**
| Grade | Market Rate | @ Market | @ Min Viable | @ Market+5pp |
|---|---|---|---|---|
| A | 9.5% | 0% (below floor) | 27% | 0% (below floor) |
| B | 14.0% | 0% (below floor) | 38% | 0% (below floor) |
| C | 19.0% | 0% (below floor) | 52% | 44% |
| D | 24.0% | 59% | 58% | 44% |
| E | 29.5% | 59% | 59% | 44% |
| F/G | 34–37% | 58% | 58% | 44% |

**Fairness note:** `zip_code` is retained as a feature. It is predictive (local economic conditions correlate with default risk) but carries disparate impact risk due to historical redlining patterns. If the agent relies heavily on this feature, it warrants discussion in the paper's limitations section.

**Cost of capital / competitive rate floor:** The simulation explicitly models a non-bank fintech lender with a high cost of capital (e.g., 16% blended warehouse + ABS funding). This imposes a minimum viable offer rate:

```
min_viable_rate = COST_OF_CAPITAL + expected_loss_provision + servicing_margin
```

For prime borrowers (Grade A/B), `min_viable_rate` substantially exceeds market rates (prime borrowers have access to cheap bank alternatives), so acceptance probability collapses even at optimal pricing. For subprime borrowers (Grade D–F), market rates are high enough that the lender is competitive. This creates a realistic market segmentation the agent must discover through iteration.

| LendingClub Grade | Approx Market Rate | Competitive for This Lender? |
|---|---|---|
| A (750+ FICO) | 8–11% | No — deeply uncompetitive |
| B (700–749) | 12–16% | No — marginally uncompetitive |
| C (670–699) | 17–21% | Borderline |
| D (640–669) | 22–26% | Yes — viable, tight margin |
| E (600–639) | 27–32% | Yes — sweet spot |
| F/G (<600) | 33%+ | Yes — high margin, high loss risk |

Cost of capital is given to the agent as an explicit known constraint (not inferred), consistent with how a real credit team would operate.

**Design notes:**
- Moderate stochasticity baked in — enough that the agent must reason in expected value terms, not enough to wash out the learning signal
- Made available as a tool the agent can call during pricing optimization
- Held fixed across all experimental variants (not re-trained between episodes)

**Portfolio constraint:** The agent operates under a **capital cap + minimum volume floor**. The cap (maximum total principal deployed) prevents unlimited cherry-picking. The volume floor (minimum number of loans funded) prevents the degenerate strategy of approving only the 5 safest loans at extreme margin, forcing the agent to navigate the full viable credit spectrum.

**Baseline parameters (tunable via `configs/base.yaml`):**
- Capital cap: **$15,000,000** per episode
- Volume floor: **400 loans** minimum per episode
- Applicant pool: **~1,000 applicants** per episode

---

## Evaluation Metrics

### Primary
- **Simulated Portfolio P&L** on validation OOT (within-loop feedback) and final holdout (reported results)
- Computed as: sum of (interest collected - principal lost) across accepted loans, weighted by acceptance probability from sensitivity model

### Secondary
- **Concordance Index (C-statistic)** on survival model
- **Acceptance Rate** — volume of offers taken
- **Total tokens consumed** per episode and per variant to plateau
- **Wall-clock latency** per episode
- **Iterations to plateau** in Ralph Wiggum loop

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
GEPA runs immediately after the baselines (Treatment 2), before any loop treatments:
- GEPA runs offline on the **training partition only**
- Optimizes the agent's prompt set (system prompts, tool descriptions, instruction modules) via reflective genetic evolution, specifically for the **35B-A3B primary model**
- Produces a locked optimized prompt set used for all subsequent treatments (T3–T7)
- GEPA is a one-time calibration step — prompts are fixed before loop evaluation begins, eliminating the confound of simultaneous prompt optimization and performance measurement
- Positioning GEPA before the loops ensures that loop gains (T3, T4) cannot be dismissed as "compensating for a suboptimal prompt" — all loop results are on optimized prompts

---

## Treatment Ladder (Ablation Study)

Each treatment is a cumulative addition to the prior level. Treatments 3–7 all run on GEPA-optimized prompts (locked after Treatment 2).

| # | Variant | Framework | Model | Loop | GEPA | Advisor | RLM STM | LT Memory |
|---|---|---|---|---|---|---|---|---|
| 0 | Vanilla ReAct | `create_agent()` | 35B-A3B | — | — | — | — | — |
| 1a | DeepAgent (batteries on) | `create_deep_agent()` | 35B-A3B | — | — | — | — | — |
| 1b | DeepAgent (batteries on) | `create_deep_agent()` | 27B | — | — | — | — | — |
| 2 | GEPA-optimized DeepAgent | `create_deep_agent()` | 35B-A3B | — | ✓ | — | — | — |
| 3 | GEPA + Loop (single-prior) | `create_deep_agent()` | 35B-A3B | Single | ✓ | — | — | — |
| 4 | GEPA + Loop (all-prior) | `create_deep_agent()` | 35B-A3B | All | ✓ | — | — | — |
| 5 | GEPA + Advisor + Loop | `create_deep_agent()` | 35B-A3B + 27B | All | ✓ | ✓ | — | — |
| 6 | GEPA + Advisor + RLM + Loop | `create_deep_agent()` | 35B-A3B + 27B | All | ✓ | ✓ | ✓ | — |
| 7 | Full System (all components) | `create_deep_agent()` | 35B-A3B + 27B | All | ✓ | ✓ | ✓ | ✓ |

**T0, T1a, T1b, and T2** are each run N=20 independent episodes (no loop) to establish performance distributions. T1a is the primary anchor for statistical significance testing on T2–T7. T1b establishes the dense model solo baseline, enabling the headline comparison: *does the full harness (T5+) on the efficient sparse MoE model surpass the dense model running alone?*

**Loop mode for T5–T7:** T3 vs T4 establishes which loop mode wins. T5–T7 use the winner (expected: all-prior). If T3 surprises, a T5a single-prior variant can be run as a follow-on.

**Key pairwise comparisons the ladder enables:**
| Comparison | What it isolates |
|---|---|
| T0 vs T1a | Full DeepAgent battery value (todos, filesystem, summarization, subagents) vs vanilla ReAct |
| T1a vs T1b | Model architecture effect (sparse MoE vs dense), identical harness |
| T1a vs T2 | Pure GEPA prompt optimization contribution |
| T2 vs T3 | Pure single-prior loop contribution, on optimized prompts |
| T3 vs T4 | All-prior vs single-prior context injection |
| T4 vs T5 | Pure advisor contribution |
| T1b vs T5 | 27B solo vs 35B-A3B + full harness — efficiency argument |
| T5 vs T6 | RLM short-term memory contribution |
| T6 vs T7 | Structured long-term memory vs naive all-prior injection |

---

## Agent Architecture

### Framework
- **T0 orchestration:** LangChain `create_agent()` (vanilla ReAct, v1.x) — no batteries, fully custom tool set
- **T1–T7 orchestration:** LangChain `create_deep_agent()` (deepagents v0.6+) — batteries included, custom supplementary tools layered on top
- **Inference:** vLLM on GX10, local OpenAI-compatible endpoint
- **Observability:** Langfuse (MIT licensed, self-hostable; first-class LangGraph integration for trace logging, cost tracking, and dataset evaluation)
- **Sandbox:** Controlled Python execution environment with filesystem access

### Agent Tools
| Tool | Present in | Description |
|---|---|---|
| `code_executor` | T0–T7 | Python sandbox with project PYTHONPATH; custom-built for our data stack |
| `filesystem_read/write` | T0 only | Custom tools with explicit path controls (replaced by deepagents built-ins in T1+) |
| `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` | T1–T7 | deepagents built-in filesystem tools (scoped to skill_dir + data/ via FilesystemPermission) |
| `execute` | T1–T7 | deepagents built-in shell execution |
| `write_todos` | T1–T7 | deepagents built-in todo list |
| `task` | T1–T7 | deepagents built-in subagent spawning |
| `sensitivity_model` | T0–T7 | Call customer acceptance model with applicant + offer features |
| `advisor_consult` | T5–T7 | Escalate to Qwen3.6-27B advisor |
| `query_long_term_memory` | T7 only | Subagent interface to Mem0 structured learning store |

### Short-Term Memory (RLM Subagent, Treatments 6-7)
A Recursive Language Model subagent with access to the full running context. Paired with aggressive compaction middleware to prevent context window bloat while preserving coherence across long episodes.

### Long-Term Memory Subagent (Treatment 7)
- **Framework:** Mem0 (open source, self-hostable), running against the local vLLM endpoint for embeddings and extraction
- **Reflection Agent:** Post-episode LangGraph node that extracts structured learning objects from the episode trace and writes to Mem0 via `client.add()`
- **Retrieval:** `query_long_term_memory` tool wraps `client.search()` — primary agent sends a natural language query, Mem0 returns semantically relevant learnings ranked by relevance
- **Handled by Mem0 internally:** Embedding generation, deduplication, temporal relevance decay, and learning clustering
- **Human review queue:** Optional gate before learnings are promoted to the long-term store (configurable)

---

## Implementation Plan

### Phase 1 — Infrastructure
- [x] Set up vLLM serving both models on GX10
- [x] Build agent scaffold: `create_agent()` (T0) and `create_deep_agent()` (T1–T7) with tool definitions
- [x] Implement controlled Python sandbox with filesystem (`code_executor` tool)
- [x] Build and calibrate synthetic customer sensitivity model (parameterize CoC floor, market rate tiers, β coefficients)
- [x] Construct survival target variables from LendingClub dataset
- [x] Implement three-way temporal data split
- [x] Build episode index logger and skill artifact storage (Ralph Wiggum loop + JSONL index)

### Phase 2 — Baselines
- [ ] Run Treatment 0 (vanilla create_agent, N=20 runs), record distribution
- [ ] Run Treatment 1a (deepagent 35B-A3B, N=20 runs), record distribution
- [ ] Run Treatment 1b (deepagent 27B, N=20 runs), record distribution

### Phase 3 — GEPA Optimization
- [ ] Integrate GEPA library or implement custom GEPA loop
- [ ] Run GEPA optimization on training partition (35B-A3B prompts only)
- [ ] Lock optimized prompts
- [ ] Run Treatment 2 (GEPA-optimized baseline, no loop)

### Phase 4 — Loop Treatments
- [ ] Implement Ralph Wiggum loop (single-prior flavor)
- [ ] Run Treatment 3
- [ ] Implement all-prior episode index injection
- [ ] Run Treatment 4

### Phase 5 — Advisor & Memory
- [ ] Implement advisor escalation pattern in LangGraph
- [ ] Run Treatment 5
- [ ] Implement RLM subagent + compaction middleware
- [ ] Run Treatment 6
- [ ] Implement long-term memory subagent (Mem0 + post-episode reflection node)
- [ ] Run Treatment 7

> **Scope note:** Treatments 1a, 1b, 2–5 are the load-bearing contributions for publication. Treatments 6–7 strengthen the paper but could be deferred to follow-on work if implementation scope becomes prohibitive. Decide at the end of Phase 4.

### Phase 5 — Evaluation & Paper
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
| T1–T7: `create_deep_agent()` | Batteries-included baseline; our ablation components layer on top. FilesystemPermission scopes built-in tools to skill_dir and data/. |
| T5–T7 use all-prior loop only | T3 vs T4 establishes which loop mode wins; subsequent treatments use the winner to keep the comparison clean. T5a (single-prior + advisor) can be run as a follow-on if interaction effects are of interest. |
| Three-way data split | Prevents adaptive overfitting of long-term memory to OOT population |
| Ralph Wiggum as universal harness | Enables cost/iteration comparison across all variants on equal footing |
| GEPA before loops (T2), not during | Eliminates confound between prompt optimization and loop performance measurement; all loop comparisons are on optimized prompts, preventing dismissal of loop gains as "compensating for bad prompts" |
| Dual baseline (T1a: 35B-A3B, T1b: 27B) | Isolates model architecture effect under identical harness; enables T1b vs T5 headline comparison: does sparse MoE + full harness beat dense model solo? |
| 35B-A3B as primary for T2–T7 | Preserves the efficiency story: sparse MoE activates ~3B params per token; advisor pattern only makes sense with a fast/cheap primary |
| Sparse MoE primary + dense advisor | Tests whether architectural diversity (not just scale) improves reasoning; advisor is only invoked selectively, keeping bulk token cost at MoE rates |
| P&L as primary metric | Grounds results in economic reality rather than model accuracy proxies |
| Capital cap + volume floor | Cap prevents cherry-picking; floor forces agent into full viable credit spectrum |
| Single-prior vs. all-prior loop | Tests local optima trapping; sharpens contrast between naive context injection and structured memory |
| Synthetic sensitivity model | LendingClub contains only funded loans — no borrower reject signal. Synthetic model avoids selection bias and provides known ground truth for analytical verification |
| Cost of capital floor | Non-bank fintech cannot competitively serve prime borrowers; CoC floor creates realistic market segmentation the agent must discover. Maps to real-world non-bank lending economics |
| Mem0 for LTM storage | Drop-in LangGraph-compatible memory layer replaces custom DynamoDB + reflection pipeline; handles embedding, deduplication, and retrieval internally |
| Langfuse for observability | MIT-licensed, self-hostable LLM observability with LangGraph integration. LangSmith requires Enterprise license for self-hosting |

---

## Open Questions & Future Work

- Capital cap and volume floor set to $15M / 400 loans as baseline — may need tuning after Treatment 1 baseline runs reveal agent behavior
- Whether GEPA should optimize per-subagent prompts independently or the full compound system
- Whether the reflection agent should run post-episode (cleaner) or on a timer (more autonomous)
- Extension to multi-agent competitive setting (multiple pricing agents competing for same borrower pool)
- Generalization test: does the full system transfer to a different credit domain (e.g., revenue-based financing for SMBs) without re-running GEPA?
- Statistical significance approach and baseline run count N — depends on observed variance in Treatment 1 distribution

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
