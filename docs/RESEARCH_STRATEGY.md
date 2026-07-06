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
- **Coverage:** 2007–2018, organized in temporal cohort files
- **Size:** ~800k+ loan records across all vintages

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
β coefficients calibrated to produce plausible acceptance behavior (30–60% average acceptance, meaningfully sensitive to rate). Calibration draws on published research on consumer loan price elasticity.

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

**Portfolio constraint:** The agent operates under a **capital cap + minimum volume floor**. The cap (maximum total principal deployed) prevents unlimited cherry-picking. The volume floor (minimum number of loans funded) prevents the degenerate strategy of approving only the 5 safest loans at extreme margin, forcing the agent to navigate the full viable credit spectrum. Exact parameter values TBD pending distribution analysis on the training partition.

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
Prior to the first Ralph Wiggum loop iteration for GEPA variants:
- GEPA runs offline on the **training partition only**
- Optimizes the agent's prompt set (system prompts, tool descriptions, instruction modules) via reflective genetic evolution
- Produces a locked optimized prompt set used for all subsequent loop iterations
- GEPA is a one-time calibration step — prompts are fixed before evaluation begins, eliminating the confound of simultaneous prompt optimization and performance measurement

---

## Treatment Ladder (Ablation Study)

Each treatment is a cumulative addition to the prior level. Every variant (except the standalone baseline) runs through both Ralph Wiggum loop flavors.

| # | Variant | Ralph Wiggum | GEPA | Advisor | RLM STM | LT Memory |
|---|---|---|---|---|---|---|
| 1 | Baseline ReAct/DeepAgent | — | — | — | — | — |
| 2 | Baseline + Loop (single-prior) | Single | — | — | — | — |
| 3 | Baseline + Loop (all-prior) | All | — | — | — | — |
| 4 | GEPA-optimized + Loop | Both | ✓ | — | — | — |
| 5 | GEPA + Advisor + Loop | Both | ✓ | ✓ | — | — |
| 6 | GEPA + Advisor + RLM + Loop | Both | ✓ | ✓ | ✓ | — |
| 7 | Full System (all components) | Both | ✓ | ✓ | ✓ | ✓ |

**Baseline (Treatment 1)** is run N times (suggested: 10-20) to establish a performance distribution. This distribution is the anchor for statistical significance testing on all subsequent treatments.

---

## Agent Architecture

### Framework
- **Orchestration:** LangGraph DeepAgent architecture
- **Inference:** vLLM on GX10, local endpoint
- **Observability:** Langfuse (MIT licensed, self-hostable; first-class LangGraph integration for trace logging, cost tracking, and dataset evaluation)
- **Sandbox:** Controlled Python execution environment with filesystem access

### Agent Tools
| Tool | Description |
|---|---|
| `code_executor` | Run Python in controlled sandbox |
| `filesystem` | Read/write artifacts, models, skill files |
| `sensitivity_model` | Call customer acceptance model with applicant + offer features |
| `query_long_term_memory` | (Treatment 7 only) Subagent interface to structured learning store |
| `advisor_consult` | (Treatments 5-7) Escalate to Qwen3.6-27B advisor |

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
- [ ] Set up vLLM serving both models on GX10
- [ ] Build LangGraph DeepAgent scaffold with tool definitions
- [ ] Implement controlled Python sandbox with filesystem
- [ ] Build and calibrate synthetic customer sensitivity model (parameterize CoC floor, market rate tiers, β coefficients)
- [ ] Construct survival target variables from LendingClub dataset
- [ ] Implement three-way temporal data split
- [ ] Build episode index logger and skill artifact storage

### Phase 2 — Baseline & Loop
- [ ] Run Treatment 1 (baseline, N=20 runs), record distribution
- [ ] Implement Ralph Wiggum loop (single-prior flavor)
- [ ] Run Treatment 2
- [ ] Implement all-prior episode index injection
- [ ] Run Treatment 3

### Phase 3 — GEPA Optimization
- [ ] Integrate GEPA library or implement custom GEPA loop
- [ ] Run GEPA optimization on training partition
- [ ] Lock optimized prompts
- [ ] Run Treatment 4 (both loop flavors)

### Phase 4 — Advisor & Memory
- [ ] Implement advisor escalation pattern in LangGraph
- [ ] Run Treatment 5
- [ ] Implement RLM subagent + compaction middleware
- [ ] Run Treatment 6
- [ ] Implement long-term memory subagent (Mem0 + post-episode reflection node)
- [ ] Run Treatment 7

> **Scope note:** Treatments 1–5 are the load-bearing contributions for publication. Treatments 6–7 strengthen the paper but could be deferred to follow-on work if implementation scope becomes prohibitive. Decide at the end of Phase 3.

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
| Three-way data split | Prevents adaptive overfitting of long-term memory to OOT population |
| Ralph Wiggum as universal harness | Enables cost/iteration comparison across all variants on equal footing |
| GEPA before loop, not during | Eliminates confound between prompt optimization and loop performance measurement |
| Sparse MoE primary + dense advisor | Tests whether architectural diversity (not just scale) improves reasoning |
| P&L as primary metric | Grounds results in economic reality rather than model accuracy proxies |
| Capital cap + volume floor | Cap prevents cherry-picking; floor forces agent into full viable credit spectrum |
| Single-prior vs. all-prior loop | Tests local optima trapping; sharpens contrast between naive context injection and structured memory |
| Synthetic sensitivity model | LendingClub contains only funded loans — no borrower reject signal. Synthetic model avoids selection bias and provides known ground truth for analytical verification |
| Cost of capital floor | Non-bank fintech cannot competitively serve prime borrowers; CoC floor creates realistic market segmentation the agent must discover. Maps to real-world non-bank lending economics |
| Mem0 for LTM storage | Drop-in LangGraph-compatible memory layer replaces custom DynamoDB + reflection pipeline; handles embedding, deduplication, and retrieval internally |
| Langfuse for observability | MIT-licensed, self-hostable LLM observability with LangGraph integration. LangSmith requires Enterprise license for self-hosting |

---

## Open Questions & Future Work

- Capital cap and volume floor exact values — needs distribution analysis on training partition to calibrate
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
