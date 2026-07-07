# Literature Review: Related Work & Novelty Assessment

*Synthesized from deep research pass across 21 sources, 5 search angles. Last updated: 2026-07-06.*

---

## Summary of Novelty

This paper occupies a gap that, as of this writing, appears unclaimed: a **controlled ablation study of agent harness components, measured by portfolio P&L on a real financial task, with a rigorous temporal holdout**. No prior work does all three of these simultaneously. The literature does self-improvement loops, does financial agent evaluation, and does memory comparisons — but never in combination, never with a clean treatment ladder that isolates each component's contribution.

---

## 1. Agent Self-Improvement Loops and Iterative Refinement

### Key Papers

**Reflexion: Language Agents with Verbal Reinforcement Learning**
Shinn, Cassano, Gopinath, Narasimhan, Yao. *NeurIPS 2023*. arXiv:2303.11366

The most structurally similar prior work to the Ralph Wiggum loop. Reflexion stores verbal self-critiques in a bounded episodic memory buffer (Ω = 1–3 most recent reflections), injected as prefix text before subsequent trial contexts. The agent improves across trials without any weight updates. Key differences from this paper: (1) Reflexion uses a *single trial per attempt*, not a full episode with multi-turn tool use; (2) the memory is a growing text accumulation, not a structured store — this is our "single-prior" and "all-prior" loop analogs; (3) Reflexion is evaluated on task-completion proxies (HumanEval pass@1, AlfWorld task success rate), not economic outcomes; (4) no ablation — the paper treats the loop as a monolithic system, not as a component among several.

**Self-Refine: Iterative Refinement with Self-Feedback**
Madaan, Tandon, Gupta, Hallinan, Gao, Green, Bosselut, Hajishirzi, Yatskar, Fried, Liu, Clark. *NeurIPS 2023*. arXiv:2303.17651

Single-LLM generate-critique-refine loop; no external oracle or second model. Closest to our Treatment 2 (single-prior loop, no advisor). Gains are task-dependent — near zero on objective math tasks (~0.2%), large on subjective generation tasks. This is a relevant caution for our paper: the loop's value depends on whether feedback is informative, which P&L feedback is.

**Gap this paper fills:** The Ralph Wiggum loop is substantively similar to Reflexion, but this paper is not primarily a loop paper — the loop is one controlled treatment in a broader ablation. More importantly, no prior work tests whether naive context injection (all-prior episode index, our Treatment 3) approximates structured long-term memory (Treatment 7) on a real economic task. Reflexion and Self-Refine never ask this question.

---

## 2. Long-Term Memory for AI Agents

### Key Papers

**MemGPT: Towards LLMs as Operating Systems**
Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez. *arXiv:2310.08560*, 2023.

OS-inspired hierarchical memory with fast (in-context) and slow (external) storage tiers. Conceptual foundation for structured long-term memory in agents. **Note:** The paper's framing of context windows as a binding constraint is partially outdated as of 2026 (1M-token windows are now standard), but the key insight — that retrieval quality degrades with naive injection at any scale — remains valid and is supported by more recent benchmarks.

**Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory**
Tariq et al. *arXiv:2504.19413*, April 2025.

This is the paper directly behind our Treatment 7 implementation. Reports 26% relative improvement in LLM-as-a-Judge scoring on the LOCOMO benchmark vs. OpenAI's full-context baseline, 91% p95 latency reduction, and >90% token savings vs. naive full-context injection. The key mechanism is selective memory extraction (not all text, only salient facts), deduplication, and semantic retrieval. This paper does not evaluate on agentic tasks requiring multi-turn economic decision-making or iterative model building.

**AMA-Bench: A Benchmark for Evaluating AI Memory in Agentic Settings**
arXiv:2507.05257, July 2025.

Recent benchmark specifically targeting agentic memory. Key finding: all tested memory methods (context, RAG, external modules, tool-integrated agents) fail on multi-hop selective forgetting tasks, with max 28% accuracy. RAG agents outperform backbone LLMs on direct retrieval but miss holistic understanding. This provides a useful reference for expected Mem0 limitations in our setting — the agent needs to retrieve causal relationships across episodes, which is exactly the failure mode AMA-Bench identifies.

**MemoryAgentBench**
arXiv:2602.22769, February 2026.

Proposes that existing memory benchmarks (including AMA-Bench, LOCOMO) are dialogue-centric and fail to evaluate the kind of memory actually needed in agentic settings: continuous interaction trajectories composed of states, actions, observations, and tool outputs. This is directly applicable to our setting — our agent's "memory" is a sequence of {risk model choice → acceptance rate → P&L outcome} trajectories, not conversational turns. Naive similarity-based retrieval systematically underperforms on causal and objective information in such trajectories. This paper motivates why our paper's Treatment 7 vs. Treatment 3 comparison is a meaningful research question: it's not yet clear which approach wins on trajectory-centric tasks.

**Gap this paper fills:** Every existing LTM evaluation uses conversational or task-completion metrics. No paper has compared naive all-prior context injection vs. structured memory retrieval on a task where the "memory" consists of iterative economic decisions and their outcomes. Our paper provides this comparison.

---

## 3. Prompt Optimization

### Key Papers

**OPRO: Large Language Models as Optimizers**
Yang, Wang, Yin, Wang, Ye, Zhou, Song. *Google DeepMind, arXiv:2309.03409*, 2023.

Foundational gradient-free prompt optimization: feed previously generated prompts and their scores as meta-prompt context; LLM generates improved candidates. Conceptual ancestor of GEPA. Key difference: OPRO generates single prompts iteratively; GEPA applies reflective evolutionary search over multi-objective Pareto fronts.

**GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning**
arXiv:2507.19457. *ICLR 2026 Oral*. UC Berkeley, Stanford, MIT, Databricks.

The paper we are using directly in Treatment 4. Merges textual reflection with multi-objective evolutionary search. Outperforms GRPO by 6% average (up to 20%) using up to 35x fewer rollouts; beats MIPROv2 by >10%, achieving +12% on AIME-2025. Applied primarily to reasoning benchmarks. Our paper is the first to apply GEPA to an economic optimization task with objective P&L outcome, and the first to use it as a one-time calibration step (locked prompts) in a controlled ablation rather than as a continuous optimization method.

**AutoPDL: Automatic Prompt Optimization for LLM Agents**
arXiv:2504.04365, April 2025.

Automatic prompt strategy selection (CoT, ReAct, Zero-Shot, etc.) via DSPy. Finds that the optimal prompting strategy varies across models and tasks — no single pattern dominates. Average gain: +9.21 pp across 3 tasks, 7 LLMs (3B–70B). Closest competitor to GEPA in the automated prompt optimization space, but does not use reflective evolutionary search, and optimizes strategy selection rather than prompt content.

**Gap this paper fills:** No prior work applies GEPA to a real economic task. GEPA's ICLR 2026 results are entirely from reasoning/math benchmarks. Our paper provides an out-of-distribution test of GEPA's effectiveness in a multi-turn, tool-use, financial modeling context.

---

## 4. Multi-Agent Supervisor/Advisor Architectures

### Key Papers

**Anthropic Advisor Tool**
*Anthropic Platform Docs, advisor_20260301 beta*, March 2026.

The direct reference for our Treatment 5 dual-model architecture. Formalizes the fast-executor / high-intelligence-advisor pattern: executor (Sonnet 4.6) handles the hot path; advisor (Opus 4.8) is invoked selectively. Designed to achieve near-advisor-solo quality while generating bulk tokens at executor rates — a favorable cost-quality tradeoff on long-horizon agentic tasks. This is a practitioner reference, not an academic paper with ablation results.

**SMoA: Sparse Mixture of Agents**
arXiv:2411.03284, November 2024.

Proposes sparse agent interaction topology — agents communicate selectively rather than in a dense fully-connected pattern. Finds that sparse interaction matches dense MoA performance at significantly lower cost, and that dense connectivity actively hampers diversity. Most relevant to our sparse MoE primary + selective advisor pattern: both architectures use controlled sparsity to manage cost. Key difference: SMoA is about agent-to-agent communication topology; our advisor pattern is about model-tier routing.

**Multi-Agent Topology and Frontier LLM Convergence**
arXiv:2602.16873, February 2026.

Shows that frontier LLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 2.0, DeepSeek-V3, Qwen 2.5 72B) now cluster within 2–5% of each other on standard benchmarks, making orchestration topology — not model selection — the dominant performance factor in multi-agent systems. This strongly supports the framing of our paper: harness design is now more important than which specific model is used, and a controlled ablation of harness components is therefore a high-value research contribution.

**Gap this paper fills:** The Anthropic Advisor Tool reference establishes the pattern but provides no academic ablation. No paper has measured the P&L contribution of adding a dual-model advisor layer, isolated from other harness changes, on a real economic task.

---

## 5. LLM Agents Applied to Financial Tasks

### Key Papers

**Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending**
Sanz-Guerrero, J. & Arroyo, J. *arXiv:2401.16458*, January 2024.

**Most directly comparable prior work.** Fine-tunes BERT on LendingClub loan descriptions to generate a default-risk score, then feeds it into XGBoost for loan-granting decisions. Improves balanced accuracy and AUC over numerical-feature-only baseline. However: (1) uses LLM as a static feature extractor, not as an agentic decision-maker; (2) evaluates on classification accuracy, not P&L; (3) no treatment ladder or harness ablation; (4) does not model borrower acceptance behavior (assumes all offers accepted). Our paper must clearly differentiate from this work in the related work section.

**LLM Agents in Stock Trading**
arXiv:2510.02209, October 2025.

Notes that static financial QA benchmarks are insufficient for evaluating LLM agents in dynamic, sequential decision-making settings. Shows that agent framework architecture drives trading performance variation more than the underlying LLM backbone — directly corroborating our paper's thesis. However, focused on equity trading (not credit), no memory treatments, no prompt optimization.

**PortBench: Evaluating LLMs for Portfolio Management**
arXiv:2605.27887, May 2026.

Finds 90% of model-profile combinations fail to beat a basic equal-weight allocation. LLMs that satisfy every procedural criterion can still miss profit-driven optimization entirely. This is a useful contrast for our paper: PortBench evaluates a single-turn or short-horizon task; our paper evaluates whether an iterative self-improvement loop can overcome baseline failures over multiple episodes.

**A Survey on LLMs for Credit Risk Assessment**
arXiv:2506.04290, June 2025.

Explicitly notes that existing LLM credit risk literature almost universally ignores compute cost and latency as evaluation metrics. LLMs offer demonstrated advantages over traditional structured-data models specifically when unstructured text signals are available. Directly supports our paper's choice to report token cost alongside P&L — this is a documented gap in the field.

**Gap this paper fills:** No prior work applies an iterative self-improvement harness to a credit pricing task. Sanz-Guerrero & Arroyo is the only paper using LendingClub with an LLM; it does not do portfolio optimization or measure P&L. Our paper is the first to (1) use an agent to iteratively construct a survival model + pricing function, (2) simulate borrower acceptance, and (3) evaluate on portfolio P&L with a temporal holdout.

---

## 6. Real-World Economic Outcome Benchmarks for Agent Evaluation

### Key Papers

**EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents**
arXiv:2503.18825, March 2025.

**Closest prior work on evaluation methodology.** Measures LLM agents on procurement, scheduling, and multi-agent pricing tasks where success is defined by actual economic outcomes (not QA accuracy). Finds existing LLM benchmarks do not capture dynamic sequential decision-making in uncertain economic environments. Most similar in spirit to our evaluation; key differences: EconEvals tasks are short-horizon (1–3 steps), not multi-turn model construction + evaluation; no iterative self-improvement; no credit domain; no treatment ladder.

**Agent Framework Architecture Dominates LLM Choice in Trading**
arXiv:2510.11695, October 2025.

Empirical study showing agent frameworks display "markedly distinct behavioral patterns" while model backbone contributes less to outcome variation. Directly validates our paper's framing: if you want to understand performance variation in financial agentic systems, study the framework, not the model. No ablation structure, no credit domain.

**Gap this paper fills:** EconEvals is the closest prior evaluation methodology, but focuses on one-shot or very short economic tasks. No paper has evaluated whether iterative self-improvement over many episodes, with a rigorous out-of-time holdout, improves LLM agent performance on an economic task requiring multi-stage model construction.

---

## 7. Sparse MoE vs. Dense Model Comparisons in Agentic Settings

### Key Papers

**Qwen3 Technical Report**
arXiv:2505.09388, Qwen Team, Alibaba Cloud, May 2025.

Establishes that sparse MoE models can match dense models at approximately 1/5 activated parameters (Qwen3-235B-A22B activates 22B of 235B total). Our primary agent (Qwen3.6-35B-A3B) activates ~3B of ~35B total parameters. The technical report includes benchmark comparisons for the model family. Does not evaluate in multi-episode agentic settings or measure compute cost vs. task performance tradeoffs.

**MoMA: Mixture of Model Agents**
arXiv 2509.07571, September 2025.

A unified routing framework using sparse mixture of models. MoMA outperforms the best single dense model on reasoning benchmarks while reducing costs by over 31%. Conceptually similar to our advisor routing pattern, but operates at the token-routing level (different model for different tokens), not the task-routing level (different model for different reasoning tasks). No credit or financial domain evaluation.

**Gap this paper fills:** No prior work evaluates a sparse MoE primary paired with a dense advisor specifically in multi-episode agentic credit pricing, nor measures the P&L uplift from adding the advisor layer in isolation. Most MoE evaluations are benchmark-based, single-pass, without iterative self-improvement.

---

## 8. Papers We Must Clearly Differentiate From

| Paper | Why it might look similar | Key differentiator |
|---|---|---|
| Sanz-Guerrero & Arroyo (2024), arXiv:2401.16458 | Uses LLM + LendingClub, credit risk | Static feature extractor, not agent; classification accuracy, not P&L; no harness |
| Reflexion (Shinn et al., NeurIPS 2023) | Iterative verbal self-improvement loop | Single-trial tasks; naive buffer injection; no ablation; no economic outcome |
| EconEvals (2025), arXiv:2503.18825 | Economic task evaluation | Short-horizon, one-shot; no iterative improvement; no treatment ladder |
| Mem0 (2025), arXiv:2504.19413 | Uses Mem0 specifically | Conversational memory, not trajectory-based; no financial task; no comparison vs. naive injection |
| GEPA (ICLR 2026), arXiv:2507.19457 | Uses GEPA directly | Math/reasoning benchmarks only; no financial application; no ablation in broader harness |

---

## Full Citation List

### Core References (all treatments)
1. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11366
2. Madaan, A., Tandon, N., et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. arXiv:2303.17651
3. Yang, C., Wang, X., et al. (2023). *Large Language Models as Optimizers (OPRO)*. arXiv:2309.03409
4. Tariq, S., et al. (2025). *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. arXiv:2504.19413
5. Zhang, Y., et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. ICLR 2026 Oral. arXiv:2507.19457
6. Anthropic. (2026). *Advisor Tool*. Claude Platform Docs, advisor_20260301 beta.

### Self-Improvement & Memory
7. Packer, C., Wooders, S., et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560
8. [Authors]. (2025). *AMA-Bench: Evaluating AI Memory in Agentic Settings*. arXiv:2507.05257
9. [Authors]. (2026). *MemoryAgentBench*. arXiv:2602.22769

### Multi-Agent Architecture
10. Li, F., et al. (2024). *SMoA: Sparse Mixture of Agents*. arXiv:2411.03284
11. [Authors]. (2026). *Multi-Agent Topology and Frontier LLM Convergence*. arXiv:2602.16873
12. [Authors]. (2025). *MoMA: Mixture of Model Agents*. arXiv:2509.07571

### Prompt Optimization
13. [Authors]. (2025). *AutoPDL: Automatic Prompt Optimization for LLM Agents*. arXiv:2504.04365

### Financial / Economic Task Evaluation
14. Sanz-Guerrero, J. & Arroyo, J. (2024). *Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending*. arXiv:2401.16458
15. [Authors]. (2025). *EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents*. arXiv:2503.18825
16. [Authors]. (2025). *LLM Agents in Stock Trading*. arXiv:2510.02209
17. [Authors]. (2025). *PortBench: Evaluating LLMs for Portfolio Management*. arXiv:2605.27887
18. [Authors]. (2025). *A Survey on LLMs for Credit Risk Assessment*. arXiv:2506.04290
19. [Authors]. (2025). *Agent Framework Architecture Drives LLM Trading Performance*. arXiv:2510.11695

### Model Infrastructure
20. Qwen Team, Alibaba Cloud. (2025). *Qwen3 Technical Report*. arXiv:2505.09388

### Consumer Credit Pricing
21. Karlan, D. & Zinman, J. (2008). *Elasticities of Demand for Consumer Credit*. Yale Economic Growth Center Discussion Paper No. 926.

---

## Recommended Related Work Section Structure (for paper)

1. **Agent Self-Improvement Loops** — Reflexion, Self-Refine, and the general question of whether iterative verbal critique improves performance. Position our loop treatments as building on Reflexion's episodic injection pattern, then testing whether structured memory (Treatment 7) outperforms it.

2. **Prompt Optimization** — OPRO as conceptual ancestor; GEPA as the specific method we apply; AutoPDL as concurrent work. Position Treatment 4 as the first application of GEPA to a real economic task.

3. **Long-Term Memory for Agents** — MemGPT for architecture motivation; Mem0 for our specific implementation; MemoryAgentBench for the open research question of trajectory-based vs. dialogue-based memory; AMA-Bench for known failure modes.

4. **Multi-Agent Supervisor Architectures** — Anthropic Advisor Tool for our specific implementation; SMoA and MoMA for the broader literature on sparse agent interaction; multi-agent topology paper for the claim that harness > model.

5. **LLM Agents in Financial Domains** — Survey the narrow prior work (Sanz-Guerrero, stock trading agents, PortBench), position our paper as the first to combine iterative self-improvement with economic P&L evaluation in a credit pricing context.

6. **Evaluation Methodology** — EconEvals as the most similar prior evaluation framework; position our temporal holdout and treatment ladder as a methodological contribution over single-shot economic agent benchmarks.
