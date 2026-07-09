from __future__ import annotations
import pathlib

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.messages import AIMessage
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from deepagents.backends.local_shell import LocalShellBackend

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
_T0_SUMMARIZE_TRIGGER   = ("tokens", 30_000)
_T0_SUMMARIZE_KEEP      = ("tokens", 10_000)
_T0_CONTEXT_EDIT_TRIGGER = 25_000
_T0_CONTEXT_EDIT_KEEP    = 5

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
                f"    sensitivity_model_defaulter.pkl, sensitivity_model_nondefaulter.pkl\n"
                f"  Working directory: {skill_dir}/\n\n"
                f"To run Python: write code with write_file, then:\n"
                f"  execute(command='python3 /absolute/path/to/script.py')\n"
                f"Never pass code= or filename= to execute — only command= is accepted."
            ),
        }
        # LocalShellBackend gives the agent real filesystem access and a
        # working execute() tool. The default SandboxBackendProtocol silently
        # returns [] for all ls() calls and blocks execute() entirely.
        # FilesystemPermission is incompatible with LocalShellBackend (deepagents
        # limitation), so we rely on system-prompt guidance instead of tool-level
        # path restrictions — acceptable for a single-user research environment.
        # 180s execute timeout: forces agent toward faster approaches (LR, sampling)
        # rather than burning 600s on full-dataset GBM/grid-search that repeatedly times out.
        backend = LocalShellBackend(root_dir=str(skill_dir), virtual_mode=False, inherit_env=True, timeout=180)
        guardrails = HarnessGuardrailsMiddleware(
            skill_dir=skill_dir,
            data_dir=PROJECT_ROOT / "data" / "processed",
        )
        results_guard = ResultsGuardMiddleware(skill_dir=skill_dir, max_retries=1)
        time_aware = TimeAwarenessMiddleware(max_seconds=treatment_config.max_episode_seconds)
        graph = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            subagents=[subagent],
            backend=backend,
            middleware=[guardrails, results_guard, time_aware],
        )

    return graph, callbacks
