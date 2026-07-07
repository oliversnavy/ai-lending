from __future__ import annotations
import pathlib

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from deepagents.graph import create_deep_agent
from deepagents import FilesystemPermission

from agent.app.clients.langfuse import get_langfuse_callback
from agent.app.clients.vllm import get_primary_client
from agent.app.models.episode import TreatmentConfig
from agent.app.prompts.gepa import get_optimised_prompt
from agent.app.prompts.system import get_system_prompt
from agent.app.tools import get_t0_tools, get_supplementary_tools

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
        # T0: vanilla create_agent — our full custom tool set, no batteries
        tools = get_t0_tools(treatment_config, skill_dir)
        graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
    else:
        # T1+: create_deep_agent with batteries on + our supplementary tools
        tools = get_supplementary_tools(treatment_config)
        permissions = [
            FilesystemPermission(
                operations=["read", "write"],
                paths=[str(skill_dir)],
                mode="allow",
            ),
            FilesystemPermission(
                operations=["read"],
                paths=[str(PROJECT_ROOT / "data" / "processed")],
                mode="allow",
            ),
        ]
        graph = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            permissions=permissions,
        )

    return graph, callbacks
