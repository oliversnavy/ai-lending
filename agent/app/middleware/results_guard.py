"""
ResultsGuardMiddleware — re-triggers the agent if results.json is missing.

When the agent finishes an episode without writing results.json (context
exhaustion, getting stuck in analysis, etc.), after_agent() fires, injects a
short recovery HumanMessage, and jumps back to the model node to restart
the agentic loop. The agent sees its full message history + pipeline files
in skill_dir and should be able to write the output in one more pass.

Max one retry per episode (max_retries=1) to avoid infinite loops.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)


class ResultsGuardMiddleware(AgentMiddleware):
    """Re-triggers the agent with a recovery prompt if results.json is absent."""

    def __init__(self, skill_dir: pathlib.Path, max_retries: int = 1) -> None:
        self.skill_dir = skill_dir
        self.max_retries = max_retries
        self._attempts = 0

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        results_path = self.skill_dir / "results.json"

        if results_path.exists():
            return None

        if self._attempts >= self.max_retries:
            log.warning(
                "[ResultsGuard] results.json still missing after %d retries — giving up",
                self._attempts,
            )
            return None

        self._attempts += 1
        log.warning(
            "[ResultsGuard] results.json not found — injecting recovery prompt (attempt %d/%d)",
            self._attempts,
            self.max_retries,
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "IMPORTANT: You completed your analysis but did not write results.json. "
                        "This file is required to record the episode result.\n\n"
                        "Do NOT re-train any models. Using the pipeline work you have already "
                        "done in this session, compute the P&L metrics and write results.json "
                        "to your working directory right now. "
                        "A rough result is far better than no result.\n\n"
                        "Required format:\n"
                        '{\n'
                        '  "pnl":             <float>,\n'
                        '  "c_stat":          <float — use Harrell\'s C, not roc_auc_score:\n'
                        '                      from lifelines.utils import concordance_index\n'
                        '                      c_stat = concordance_index(val["observed_time"], -pred_prob, val["event"])>,\n'
                        '  "acceptance_rate": <float>,\n'
                        '  "loans_funded":    <int>,\n'
                        '  "total_principal": <float>,\n'
                        '  "approach":        "<one sentence>",\n'
                        '  "hypothesis":      "<one sentence>"\n'
                        "}"
                    )
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
