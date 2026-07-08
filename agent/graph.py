from __future__ import annotations
import pathlib

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from deepagents.backends.local_shell import LocalShellBackend

from agent.app.clients.langfuse import get_langfuse_callback
from agent.app.clients.vllm import get_primary_client
from agent.app.models.episode import TreatmentConfig
from agent.app.prompts.gepa import get_optimised_prompt
from agent.app.prompts.system import get_system_prompt
from agent.app.tools import get_t0_tools, get_supplementary_tools

# Trigger summarization at 30K tokens, keep the most recent 10K.
# With 65K context, 8K thinking budget, and 8K max output, effective
# input headroom is ~49K — 30K trigger leaves a comfortable margin.
_T0_SUMMARIZE_TRIGGER = ("tokens", 30_000)
_T0_SUMMARIZE_KEEP    = ("tokens", 10_000)

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
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[summarizer],
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
                f"    train.parquet, val.parquet, holdout.parquet, sensitivity_model.pkl\n"
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
        backend = LocalShellBackend(root_dir=str(skill_dir), virtual_mode=False, inherit_env=True)
        graph = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            subagents=[subagent],
            backend=backend,
        )

    return graph, callbacks
