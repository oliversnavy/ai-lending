from __future__ import annotations
import pathlib

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.messages import AIMessage
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent

from agent.app.backends.sandboxed_shell import SandboxedLocalShellBackend
from agent.app.clients.langfuse import get_langfuse_callback
from agent.app.clients.vllm import get_primary_client
from agent.app.models.episode import TreatmentConfig
from agent.app.prompts.gepa import get_optimised_prompt
from agent.app.prompts.system import get_system_prompt
from agent.app.middleware.guardrails import HarnessGuardrailsMiddleware
from agent.app.middleware.overflow_recovery import OverflowRecoveryMiddleware
from agent.app.middleware.results_guard import ResultsGuardMiddleware
from agent.app.middleware.time_awareness import TimeAwarenessMiddleware
from agent.app.tools import get_t0_tools, get_supplementary_tools

# Context management constants for T0 (vanilla ReAct).
# Context limit = 65K total. With 8K thinking budget reserved, safe input = 57K.
# Approximate token counter undercounts code-heavy content by ~2x vs real tokens.
#
# SummarizationMiddleware (before_model, modifies LangGraph state):
#   Fires at 30K approx ≈ 60K real — catches sustained long-running episodes.
#   Keeps 10K approx ≈ 20K real of recent context after compression.
#
# ContextEditingMiddleware (wrap_model_call, modifies only the API request):
#   Last-line-of-defence: fires at 25K approx ≈ 50K real, leaving a 7K buffer
#   before the hard 57K limit. Replaces old tool results with "[cleared]" in
#   the request only — LangGraph state is untouched, so Summarization still
#   compresses properly on the next turn. Keeps the 5 most recent tool results.
#
# These stay tuned to the *old* 65,536 vllm-primary max-model-len even though
# the server itself now runs at 131,072 (see _DEEP_AGENT_CONTEXT_EDIT_TRIGGER
# below) -- T0 already completed its target 20-valid-episode run under this
# budget, so there's no reason to touch it; a smaller-than-necessary trigger
# just means it compacts a bit earlier than it strictly needs to, not a bug.
_T0_SUMMARIZE_TRIGGER   = ("tokens", 30_000)
_T0_SUMMARIZE_KEEP      = ("tokens", 10_000)
_T0_CONTEXT_EDIT_TRIGGER = 25_000
_T0_CONTEXT_EDIT_KEEP    = 5

# Context management constants for T1+ (deep agent).
# 2026-07-12: vllm-primary's --max-model-len raised from 65,536 to 131,072
# (vllm-advisor stopped to free the memory; see docs/infra_notes.md and the
# 2026-07-12 project memory entry for the full incident/decision writeup).
# This was the actual fix for a night of repeated "maximum context length is
# 65536 tokens" crashes in T1a: ContextEditingMiddleware's approximate token
# counter was confirmed (via live instrumentation) to undercount T1a's dense
# numeric tool output by ~2.5x -- worse than the ~2x factor T0's own trigger
# assumes -- so no amount of threshold-tuning against the old 65,536 ceiling
# was reliable. Doubling the real ceiling makes that undercount tolerable
# instead of trying to out-guess it.
# Trigger derivation: effective input budget is now 131,072 - 8,192 (primary
# output reserve) = 122,880 real tokens. At a conservative worst-case 2.5x
# undercount, an approximate-count trigger needs to stay under roughly
# 122,880 / 2.5 ≈ 49,152 to guarantee firing before the real hard limit;
# 40,000 leaves comfortable margin below that without giving up most of the
# new headroom (a smaller intermediate step vs. reusing T0's 25,000 trigger,
# which would barely use any of the doubled context at all).
_DEEP_AGENT_CONTEXT_EDIT_TRIGGER = 40_000
_DEEP_AGENT_CONTEXT_EDIT_KEEP    = 5

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending")


def _resolve_system_prompt(tc: TreatmentConfig) -> str:
    if tc.use_gepa_prompts:
        return get_optimised_prompt() or get_system_prompt(tc.use_advisor)
    return get_system_prompt(tc.use_advisor)


def build_graph(treatment_config: TreatmentConfig, skill_dir: pathlib.Path):
    """Return (compiled_graph, callbacks) for one episode."""
    llm = get_primary_client(treatment_config)
    system_prompt = _resolve_system_prompt(treatment_config)
    callbacks = [cb for cb in [get_langfuse_callback()] if cb]

    if not treatment_config.use_deep_agent:
        # T0: vanilla create_agent — our full custom tool set, no batteries.
        # SummarizationMiddleware is the one exception: without it the
        # growing tool-call history overflows the 65K context window mid-episode.
        # deepagents includes this by default; we add it manually here so T0
        # and T1a both have context management and the comparison stays fair.
        tools = get_t0_tools(treatment_config, skill_dir)
        summarizer = SummarizationMiddleware(
            model=llm,
            trigger=_T0_SUMMARIZE_TRIGGER,
            keep=_T0_SUMMARIZE_KEEP,
        )
        context_editor = ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=_T0_CONTEXT_EDIT_TRIGGER,
                keep=_T0_CONTEXT_EDIT_KEEP,
            )],
        )
        overflow_recovery = OverflowRecoveryMiddleware()
        time_aware = TimeAwarenessMiddleware(max_seconds=treatment_config.max_episode_seconds)
        # Middleware ordering matters for wrap_model_call: last = innermost wrapper.
        # overflow_recovery is last so it sits closest to the actual API call and
        # catches real 400 errors before context_editor or summarizer see them.
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[summarizer, context_editor, time_aware, overflow_recovery],
        )
    else:
        # T1+: create_deep_agent with batteries on + our supplementary tools
        tools = get_supplementary_tools(treatment_config)
        # Override the default general-purpose subagent with explicit paths
        # so spawned subagents know where to find data without searching.
        subagent: SubAgent = {
            "name": "general-purpose",
            "description": (
                "General-purpose Python/data-science assistant. "
                "Use for data exploration, model training, evaluation, or any "
                "task that benefits from running code in a subprocess."
            ),
            "system_prompt": (
                f"You are a Python/data-science assistant helping with a credit risk task.\n\n"
                f"Key paths (always use absolute paths in code):\n"
                f"  Data (read-only):  {PROJECT_ROOT / 'data' / 'processed'}/\n"
                f"    train.parquet, val.parquet, holdout.parquet\n"
                f"    val_features_only.parquet, holdout_features_only.parquet (event/observed_time removed --\n"
                f"      this is what the harness re-scores your saved artifacts against, not the files above)\n"
                f"    sensitivity_model_defaulter.pkl, sensitivity_model_nondefaulter.pkl\n"
                f"  Working directory: {skill_dir}/\n\n"
                f"DELIVERABLE (if you're asked to build/finalize a model or pricing logic): the episode's\n"
                f"output is TWO files saved to the working directory, independently reloaded and re-scored\n"
                f"by a separate harness process after the episode ends -- nothing you compute/print yourself\n"
                f"is used for the final P&L/C-stat:\n"
                f"  risk_model.pkl    -- pickled object exposing predict_default_proba(X: pd.DataFrame) -> np.ndarray\n"
                f"  pricing_policy.py -- defines price(X: pd.DataFrame, p_default_hat: np.ndarray,\n"
                f"                       required_rate: np.ndarray) -> np.ndarray (NaN = decline)\n"
                f"Pickling gotcha: if you define your model class in a script run directly, Python records its\n"
                f"module as __main__ and the harness's separate process can't unpickle it. Put the class in its\n"
                f"own file and import it before pickling.\n\n"
                f"To run Python: write code with write_file, then:\n"
                f"  execute(command='python3 /absolute/path/to/script.py')\n"
                f"Never pass code= or filename= to execute — only command= is accepted."
            ),
        }
        # SandboxedLocalShellBackend gives the agent real filesystem access and a
        # working execute() tool, but execute() runs inside an ephemeral Docker
        # container: PROJECT_ROOT is bind-mounted read-only, skill_dir read-write
        # on top, --network none. This replaced plain LocalShellBackend after a
        # real incident (2026-07-11, T1a episode_0002): with an unsandboxed
        # LocalShellBackend, shell commands run directly on the host with no path
        # restriction at all (execute() bypasses HarnessGuardrailsMiddleware's
        # write-path check, which only applies to the write_file/edit_file native
        # tools), and the episode used `cat > .../data_pipeline/sensitivity_model.py
        # << EOF` to overwrite a shared project file. See sandboxed_shell.py for
        # the full incident writeup and design rationale.
        # 180s execute timeout: forces agent toward faster approaches (LR, sampling)
        # rather than burning 600s on full-dataset GBM/grid-search that repeatedly times out.
        backend = SandboxedLocalShellBackend(
            root_dir=str(skill_dir),
            virtual_mode=False,
            inherit_env=True,
            timeout=180,
            project_root=PROJECT_ROOT,
        )
        guardrails = HarnessGuardrailsMiddleware(
            skill_dir=skill_dir,
            data_dir=PROJECT_ROOT / "data" / "processed",
        )
        results_guard = ResultsGuardMiddleware(skill_dir=skill_dir, max_retries=1)
        time_aware = TimeAwarenessMiddleware(max_seconds=treatment_config.max_episode_seconds)
        # create_deep_agent() unconditionally injects its own SummarizationMiddleware
        # via deepagents.middleware.summarization.compute_summarization_defaults().
        # That function only picks a sane fraction-based trigger when the model has
        # LangChain profile metadata exposing max_input_tokens -- our locally-hosted
        # vLLM model has none, so it falls back to a fixed trigger=("tokens", 170_000).
        # That's still above the real hard limit even after the 2026-07-12 context
        # expansion (131,072), so this fallback default remains permanently inert --
        # ContextEditingMiddleware below is the only thing actually protecting T1a/T1b.
        # There's no public create_deep_agent() hook to exclude the auto-injected
        # default, so this adds the same stateless, request-only backstop T0 uses
        # (ContextEditingMiddleware operates on the outgoing request only, not
        # LangGraph state, so it can't conflict with the auto-injected summarizer --
        # it just clears old tool results well before either one would ever need to).
        # Positioned last so it's innermost, closest to the actual API call.
        context_editor = ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=_DEEP_AGENT_CONTEXT_EDIT_TRIGGER,
                keep=_DEEP_AGENT_CONTEXT_EDIT_KEEP,
            )],
        )
        graph = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            subagents=[subagent],
            backend=backend,
            middleware=[guardrails, results_guard, time_aware, context_editor],
        )

    return graph, callbacks
