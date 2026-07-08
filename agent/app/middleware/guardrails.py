"""
Guardrail middleware for T1+ deepagents runs.

Intercepts write_file, edit_file, and execute tool calls before they run and
blocks anything that would:
  - Touch files outside the allowed directories (skill_dir, data/processed/)
  - Run destructive or network shell commands (rm, curl, git push, etc.)

Returns a structured error ToolMessage so the model can self-correct without
the harness crashing.
"""
from __future__ import annotations

import logging
import pathlib
import re
from typing import Any, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

log = logging.getLogger(__name__)

# Shell patterns that are never allowed, regardless of path.
# Note: bare `rm` (single file) within skill_dir is handled separately in
# _check_command() and is allowed. Only recursive/force deletes are caught here.
_BLOCKED_SHELL_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[rRfFiI]*[rR][rRfFiI]*\b", "rm -r/-rf is not allowed — only single-file rm within your working directory is permitted"),
    (r"\brmdir\b", "rmdir is not allowed"),
    (r"\bcurl\b", "network access via curl is not allowed"),
    (r"\bwget\b", "network access via wget is not allowed"),
    (r"\bpip\b.*install", "pip install is not allowed during an episode"),
    (r"\bsudo\b", "sudo is not allowed"),
    (r"\bgit\s+(push|reset|checkout|rebase|merge|branch\s+-[Dd])", "destructive git operations are not allowed"),
    (r"\bchmod\b", "chmod is not allowed"),
    (r"\bchown\b", "chown is not allowed"),
    (r"\bkill\b|\bpkill\b|\bkillall\b", "killing processes is not allowed"),
    (r"\bcrontab\b", "crontab modification is not allowed"),
    (r"\bdd\b\s+if=", "dd is not allowed"),
    (r">\s*/dev/sd[a-z]", "writing to raw block devices is not allowed"),
]

# File-write tool names in deepagents
_WRITE_TOOL_NAMES = {"write_file", "edit_file"}

# Shell execution tool names
_EXEC_TOOL_NAMES = {"execute"}


def _violation_message(tool_name: str, reason: str) -> ToolMessage:
    return ToolMessage(
        content=(
            f"GUARDRAIL BLOCKED [{tool_name}]: {reason}\n"
            "Correction guidance:\n"
            "- To modify an existing file: use edit_file, not rm + write_file\n"
            "- To write new files: use write_file with a path inside your working directory\n"
            "- To run shell commands: only python3, standard data-science tools, and read operations are allowed"
        ),
        tool_call_id="__guardrail__",
        status="error",
    )


class HarnessGuardrailsMiddleware(AgentMiddleware):
    """
    Blocks file writes outside allowed directories and dangerous shell commands.

    Args:
        skill_dir: Episode working directory — the only place the agent may write.
        data_dir:  Read-only data directory — listed so ls() works, but write blocked.
    """

    def __init__(self, skill_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
        self._skill_dir = skill_dir.resolve()
        self._allowed_write = str(self._skill_dir)
        self._data_dir = str(data_dir.resolve())

    # ------------------------------------------------------------------
    # Synchronous path (used by graph.stream / graph.invoke)
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        violation = self._check(request)
        if violation:
            tool_name = request.tool_call.get("name", "unknown")
            log.warning("[guardrail] blocked %s: %s", tool_name, violation)
            msg = _violation_message(tool_name, violation)
            # patch the tool_call_id so LangGraph can match it
            msg.tool_call_id = request.tool_call.get("id", "__guardrail__")
            return msg
        return handler(request)

    # ------------------------------------------------------------------
    # Async path (used by astream / ainvoke)
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        violation = self._check(request)
        if violation:
            tool_name = request.tool_call.get("name", "unknown")
            log.warning("[guardrail] blocked %s: %s", tool_name, violation)
            msg = _violation_message(tool_name, violation)
            msg.tool_call_id = request.tool_call.get("id", "__guardrail__")
            return msg
        return await handler(request)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check(self, request: ToolCallRequest) -> str | None:
        """Return a violation reason string, or None if the call is allowed."""
        name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        if name in _WRITE_TOOL_NAMES:
            return self._check_write_path(args.get("file_path") or args.get("path", ""))

        if name in _EXEC_TOOL_NAMES:
            return self._check_command(args.get("command", ""))

        return None

    def _check_write_path(self, path: str) -> str | None:
        if not path:
            return None
        resolved = str(pathlib.Path(path).resolve())
        if not resolved.startswith(self._allowed_write):
            return (
                f"write to '{path}' is outside your working directory "
                f"({self._allowed_write}). Write files there instead."
            )
        return None

    def _check_command(self, command: str) -> str | None:
        # Special case: bare `rm <path>` — allow within skill_dir, block outside.
        # This lets the agent delete its own working files (e.g., to replace a script)
        # without being able to touch harness code or system files.
        bare_rm = re.search(r'\brm\s+(?!-)([\S]+)', command)
        if bare_rm and not re.search(r'\brm\s+-', command):
            target = bare_rm.group(1)
            if target.startswith('/'):
                resolved = str(pathlib.Path(target).resolve())
            else:
                resolved = str((self._skill_dir / target).resolve())
            if not resolved.startswith(self._allowed_write):
                return (
                    f"rm '{target}' is outside your working directory "
                    f"({self._allowed_write}). You may only delete files you created."
                )
            return None  # rm within skill_dir is allowed

        for pattern, reason in _BLOCKED_SHELL_PATTERNS:
            if re.search(pattern, command):
                return reason
        return None
