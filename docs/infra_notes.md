# Infra notes

Operational gotchas for the vLLM model servers. Not research content —
kept separate from RESEARCH_STRATEGY.md/LITERATURE_REVIEW.md.

## vllm-primary must be pinned to a specific model revision

`docker run ... vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 --revision
612d523c58522734a1f6dc995bdeebe216647fef ...` — the `--revision` and the two
`-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1` env vars are load-bearing,
not optional.

**Why:** on 2026-07-12, restarting `vllm-primary` from scratch (no revision
pin, no offline mode) crashed on load with:

```
ValueError: There is no module or parameter named 'lm_head.weight_scale' in
Qwen3_5MoeForCausalLM. The available parameters belonging to lm_head
(ParallelLMHead) are: {'lm_head.weight'}
```

Root cause: the container had been running continuously for 5 days without
ever re-resolving its model reference. A bare `--revision main`-equivalent
(no pin) triggers a fresh Hub lookup on every new container start; the
upstream `unsloth/Qwen3.6-35B-A3B-NVFP4` repo's `main` branch had moved to a
newer commit (`739af1e7...`) since the original container launched, and that
newer export adds an `lm_head.weight_scale` quantization tensor this vLLM
build's `Qwen3_5MoeForCausalLM` doesn't know how to load. A second, separate
issue hit the companion tokenizer repo (`Qwen/Qwen3.6-35B-A3B`, which has no
local `config.json` cached) for the same reason — a fresh container re-resolves
it over the network instead of trusting the local tokenizer-only cache.

The original working checkpoint (June 28 snapshot, single-file
`model.safetensors`, only `lm_head.weight` — no scale tensor) is still fully
present locally at:

```
~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4/snapshots/612d523c58522734a1f6dc995bdeebe216647fef/
```

Pinning `--revision` to that hash plus forcing offline mode (so nothing
re-resolves against the Hub at all) reproduces the exact known-good load.
**Do not drop the `--revision` pin or the offline env vars on a future
restart** — a bare restart will silently re-trigger this exact failure
(and, worse, a full ~48GB re-download of the newer incompatible revision
before failing).

If the upstream repo needs to be intentionally upgraded later, do it as a
deliberate, tested step (confirm the new revision loads cleanly before
retiring the pin), not as a side effect of a routine container restart.

## Context window / memory allocation (2026-07-12)

`vllm-advisor` (27B model, port 8001) is stopped to free memory so
`vllm-primary` (35B model, port 8000) can run with `--max-model-len 131072`
(up from 65536) and `--gpu-memory-utilization 0.65` (up from 0.35). This
fixed a real, repeated T1a failure mode — see `agent/graph.py`'s
`_DEEP_AGENT_CONTEXT_EDIT_TRIGGER` comment for the full mechanism.

Current headroom at this config: `free -h` shows ~30GB "available" with only
vllm-primary running — at the edge of the user's requested 20-30GB
reserve, so this is the final utilization figure, not bumped further.

**Downstream implications:**
- **T1b** (`primary_is_27b=True`) needs `vllm-advisor` running again before
  it can be launched — bring it back up first (`docker start vllm-advisor`,
  no config changes needed, its own `--max-model-len 32768` / memory budget
  is untouched). T1b never needs `vllm-primary` and `vllm-advisor` running
  simultaneously, so this isn't a permanent conflict.
- **T5** (`use_advisor=True`, dual-model architecture) *does* need both
  models resident at once. Revisit the memory budget for both when T5 is
  actually being built — likely means shrinking `vllm-primary` back down
  from 131072, or finding room for a smaller combined footprint. Not an
  immediate concern; T5 is far down the ablation ladder.
