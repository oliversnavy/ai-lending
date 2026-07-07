from __future__ import annotations
import pathlib

from langchain_core.tools import BaseTool

from ..models.episode import TreatmentConfig
from .code_executor import build_code_executor
from .filesystem import build_filesystem_tools
from .sensitivity_model import sensitivity_model_query


def get_t0_tools(treatment_config: TreatmentConfig, skill_dir: pathlib.Path) -> list[BaseTool]:
    """Full tool set for T0 (vanilla create_agent — no deepagents built-ins)."""
    fs_read, fs_write = build_filesystem_tools(skill_dir)
    return [
        build_code_executor(skill_dir),
        fs_read,
        fs_write,
        sensitivity_model_query,
    ]


def get_supplementary_tools(treatment_config: TreatmentConfig) -> list[BaseTool]:
    """
    Supplementary tools added on top of create_deep_agent built-ins (T1+).

    deepagents provides: ls, read_file, write_file, edit_file, glob, grep,
    execute, write_todos, task. We add only what it doesn't provide.
    """
    tools: list[BaseTool] = [sensitivity_model_query]

    if treatment_config.use_advisor:
        from ..clients.vllm import get_advisor_client
        from .advisor_consult import build_advisor_consult_tool
        tools.append(build_advisor_consult_tool(get_advisor_client()))

    if treatment_config.use_ltm:
        from .long_term_memory import query_long_term_memory
        tools.append(query_long_term_memory)

    return tools
