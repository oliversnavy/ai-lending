# Pilot Episodes

These episodes were run during harness development (2026-06-28 to 2026-07-08) to
validate and debug the agent infrastructure. They are archived here rather than
discarded because they document the evolution of the harness and contain useful
signals for the paper's methods section.

**These episodes should NOT be included in any treatment comparison.** They were
run under varying harness configurations and do not satisfy the controlled
conditions of the final experimental design.

---

## Harness Freeze (2026-07-08)

The harness was declared frozen after T1a ep0005. Settled configuration:

| Component | Value |
|---|---|
| Primary model | `unsloth/Qwen3.6-35B-A3B-NVFP4` on port 8000 |
| T1b model | `unsloth/Qwen3.6-27B-NVFP4` on port 8001 |
| Context window | 65536 (35B) / 32768 (27B) |
| thinking_budget | 8192 tokens |
| T0 middleware | `[SummarizationMiddleware(30K→10K), TimeAwarenessMiddleware(3600s)]` |
| T1+ middleware | `[HarnessGuardrailsMiddleware, ResultsGuardMiddleware, TimeAwarenessMiddleware(3600s)]` |
| T1+ execute timeout | 180s (forces fast approaches; GBM at scale times out, agent pivots to LR) |
| APR constraint | 21–36% hard cap in system prompt (prevents reward hacking) |
| `rm` policy | Single-file `rm` within skill_dir allowed; `rm -rf` and paths outside skill_dir blocked |

---

## T0 Pilot Episodes (vanilla `create_agent`, 35B-A3B)

### ep0000 — $9,235,767 | C-stat 0.699

**Harness at time of run:** `SummarizationMiddleware` only (no `TimeAwarenessMiddleware`).

**Outcome:** Valid result. Agent built a gradient-boosting default model (C-index 0.699),
applied risk-adjusted rates (`22% + 40% × default_prob`), filtered to Grade C–F.
Acceptance 34.1%, 724 loans funded, $14.99M principal.

**Key learning:** T0 is capable of producing a reasonable baseline in one shot.
The 34.1% acceptance rate reflects the 21–36% APR constraint correctly applied —
agent discovered that prime (A/B) borrowers reject at these rates.

**Why not included in paper sample:** Missing `TimeAwarenessMiddleware`. Future T0
episodes have wall-clock warnings at 50/75/90% — ep0000 did not. Runtime was not
recorded but did not appear problematic for T0's simpler tool set.

---

### ep0001 — INVALID (reward hacking)

**Harness at time of run:** Same as ep0000.

**Outcome:** Agent offered 79% APR rates across the board. P&L inflated by the
pricing formula (interest income scales with offered rate). No meaningful credit
model built.

**Root cause:** The system prompt did not specify an APR ceiling. The agent
discovered it could maximize `offered_rate × observed_time × principal` without
defaulting to realistic market rates.

**Fix applied:** Added explicit `## Rate Constraints` section to system prompt:
- Offered rate must be between 21% and 36% APR
- Rates below 21% do not cover cost of capital; rates above 36% are predatory
  and legally prohibited in the target market segment

This fix was applied before any further episodes.

---

## T1a Pilot Episodes (`create_deep_agent`, 35B-A3B)

### ep0000 — Killed (~75 min, no result)

**Harness at time of run:** Default deepagents `SandboxBackendProtocol`.

**Outcome:** Agent spawned a task subagent to run Python, but the sandbox backend
silently returned `[]` for all `ls()` calls and blocked `execute()` entirely. The
subagent, receiving empty directory listings, generated synthetic LendingClub-like
data from memory rather than reading the actual parquet files. The resulting model
was trained on hallucinated data. Episode was killed when the pattern was identified.

**Root cause:** deepagents' default backend (`SandboxBackendProtocol`) is designed
for isolated sandboxed execution. In our setup (local research box), it provides no
real filesystem access.

**Fix applied:** `LocalShellBackend(root_dir=str(skill_dir), virtual_mode=False,
inherit_env=True, timeout=180)` — gives the agent real filesystem access and a
working `execute()` tool.

**Secondary issue discovered:** `FilesystemPermission` is incompatible with
`LocalShellBackend` (deepagents raises immediately). Path restrictions replaced by
system-prompt guidance + `HarnessGuardrailsMiddleware`.

---

### ep0001 — Killed (~20 min, no result)

**Harness at time of run:** Partial fix — `LocalShellBackend` not yet applied correctly.

**Outcome:** Same synthetic-data issue as ep0000. Killed after confirming the fix
approach (switch to `LocalShellBackend`) and implementing it.

---

### ep0002 — Crashed (<1 second, `TypeError`)

**Harness at time of run:** First attempt at `LocalShellBackend` + custom subagent.

**Error:**
```
TypeError: create_deep_agent() got an unexpected keyword argument 'profile'
```

**Root cause:** Initial attempt used a `HarnessProfile` / `GeneralPurposeSubagentProfile`
approach that is not a valid argument to `create_deep_agent()`. The deepagents API
does not expose a `profile=` parameter.

**Fix applied:** Use `subagents=[SubAgent(...)]` with `name="general-purpose"` to
override the built-in subagent's system prompt with explicit data paths:
```python
subagent: SubAgent = {
    "name": "general-purpose",
    "system_prompt": f"Key paths:\n  Data: {data_dir}/\n  Working dir: {skill_dir}/\n..."
}
```

---

### ep0003 — No result (55 min, 1,405,336 tokens, context exhausted)

**Harness at time of run:** `LocalShellBackend` working, correct subagent paths. No
`HarnessGuardrailsMiddleware`, no `ResultsGuardMiddleware`, no `TimeAwarenessMiddleware`.

**Outcome:** Agent had real data access and real code execution. It successfully
loaded the parquet files, ran Python scripts, and trained a default model. However,
it spent the full 55 minutes iterating on model performance and ran out of tool-call
turns before writing `results.json`.

**Key learnings:**
1. **Agent has no wall-clock awareness.** Without explicit time injection, the agent
   treats every iteration as if it has unlimited time.
2. **The `results.json` contract needs enforcement.** The agent finished its model work
   but never translated results into the required output format.
3. **Write path discipline needed.** Agent wrote some files to the project root
   (`/home/.../ai-lending/pipeline.py`) instead of `skill_dir`.

**Fixes applied:**
- `TimeAwarenessMiddleware`: injects `HumanMessage` at 50%, 75%, 90% of budget
- `ResultsGuardMiddleware`: fires `after_agent()` with recovery prompt if
  `results.json` is absent; re-triggers via `jump_to="model"` (max 1 retry)
- `HarnessGuardrailsMiddleware`: blocks writes outside `skill_dir`, blocks `rm`,
  `curl`, `sudo`, `pip install`, and other dangerous shell commands
- Added `## Working Directory` and `## Time Budget` sections to system prompt
- Added `## Rate Constraints` reinforcement

---

### ep0004 — $19,096,161 | C-stat 0.673 | 123 min | 1,659,144 tokens

**Harness at time of run:** `LocalShellBackend` (600s timeout), `HarnessGuardrailsMiddleware`,
`ResultsGuardMiddleware`. **Missing:** `TimeAwarenessMiddleware` (not yet deployed).

**Outcome:** First completed T1a episode with a valid `results.json`. Agent built a
logistic regression default model (C-index 0.673), used risk-based pricing with a
grid-searched multiplier (~41). Acceptance 53.9%, 458 loans, $15M principal.

**Key observations:**
- **P&L $19.1M vs T0's $9.2M.** The gap is plausible given the pricing formula's
  reliance on retrospective `observed_time` (see Note on P&L Validity below).
- **Runtime problem.** Three separate GBM/grid-search scripts (`optimize.py`,
  `optimize2.py`, `final_opt.py`, `final_tune.py`) each timed out at 600s before
  the agent simplified to logistic regression. 40+ minutes were wasted on timeout
  loops.
- `rm` blocked correctly by guardrails when agent tried to overwrite `pipeline.py`.
  Agent worked around by creating `pipeline_v2.py` — a valid workaround.

**Fixes applied:**
- `LocalShellBackend(timeout=180)` — 3-minute cap forces the agent toward fast
  approaches (LR trains in ~2s; GBM at scale times out and the agent pivots)
- `TimeAwarenessMiddleware` deployed (active from ep0005 onward)

---

### ep0005 — $21,224,206 | C-stat 0.653 | 54.5 min | 830,216 tokens

**Harness at time of run:** First full middleware stack:
`[HarnessGuardrailsMiddleware, ResultsGuardMiddleware, TimeAwarenessMiddleware]`
+ `LocalShellBackend(timeout=180)`. **Missing:** `rm`-within-skill_dir fix (deployed
after this run completed).

**Outcome:** Logistic regression on 200K subsampled training rows. Per-grade optimal
rate selection (21–36%) via sensitivity model grid search. C-F grades only.
Acceptance 46.9%, 719 loans, $15M principal.

**Key observations:**
- **Runtime halved** vs ep0004 (54.5 min vs 123 min). Token use halved (830K vs 1.66M).
  `TimeAwarenessMiddleware` 75% warning (fired at 47 min: "13 min remaining, write
  results.json NOW") appears to have triggered the agent's final submission at 54.5 min.
- **`ResultsGuardMiddleware` did not fire** — agent submitted `results.json` independently.
- **`rm` block caused version proliferation.** Agent hit `write_file`'s "file already
  exists" error, tried `rm pipeline.py` (blocked), then created `pipeline_v2.py` through
  `pipeline_v6.py`, `run.py` through `run3.py`. Resolved by allowing bare `rm` within
  `skill_dir` (next fix).
- Agent noted P&L of $21M "seems suspicious" — correctly identified the `observed_time`
  inflation issue in its own `model→done` message.

**Fix applied (post-run):**
- `HarnessGuardrailsMiddleware`: bare `rm <path>` within `skill_dir` now allowed.
  `rm -rf`, `rm -r`, and `rm` targeting paths outside `skill_dir` remain blocked.

---

## Note on P&L Validity

The P&L formula uses retrospective `observed_time` from the validation set. The 2015–2016
LendingClub loans that did not default have up to 60 months of actual observation, meaning
the interest accrual in the formula reflects 5 years of payments. In real deployment, the
agent would price at origination with uncertain future tenure.

**Impact:** Absolute P&L figures in pilot episodes (and in the experiment) are inflated
relative to realistic deployment. However, this bias is **consistent across all treatments**
— the formula is identical — so relative comparisons between treatments remain valid.

An external evaluator (`evaluate.py`) using a fixed observed tenure (e.g., 24 months) would
produce more realistic absolutes. This is noted as future work; it does not affect the
treatment comparison.

---

## Infrastructure Evolution Summary

| Change | Applied Before |
|---|---|
| `LocalShellBackend` (real filesystem + execute) | T1a ep0003 |
| `SubAgent` with explicit data paths | T1a ep0003 |
| 21–36% APR cap in system prompt | T1a ep0003 (from T0 ep0001 fix) |
| `HarnessGuardrailsMiddleware` | T1a ep0005 |
| `ResultsGuardMiddleware` | T1a ep0005 |
| `TimeAwarenessMiddleware` | T1a ep0005 |
| `LocalShellBackend(timeout=180)` | T1a ep0005 |
| `## Working Directory` + `## Time Budget` in system prompt | T1a ep0005 |
| `rm` within skill_dir allowed (bare single-file only) | **Harness freeze (ep0006+)** |
