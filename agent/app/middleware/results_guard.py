"""
ResultsGuardMiddleware — re-triggers the agent if its deliverable artifacts are missing.

When the agent finishes an episode without saving risk_model.pkl + pricing_policy.py
(context exhaustion, getting stuck in analysis, etc.), after_agent() fires, injects a
short recovery HumanMessage, and jumps back to the model node to restart the agentic
loop. The agent sees its full message history + pipeline files in skill_dir and should
be able to save the artifacts in one more pass.

Deliberately does NOT ask the agent to compute or report final metrics itself — pnl/
c_stat/etc. are computed independently by agent/evaluator/evaluate_episode.py from the
submitted artifacts. Earlier prompt language here ("compute the P&L metrics and write
results.json right now") is what nudged episode_0005 toward reporting fabricated
numbers under time pressure; the fix is structural (the agent's self-report is no
longer authoritative for anything but free-text approach/hypothesis), not just a
wording change, but the wording still shouldn't invite the old failure mode.

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
    """Re-triggers the agent with a recovery prompt if deliverable artifacts are absent."""

    def __init__(self, skill_dir: pathlib.Path, max_retries: int = 1) -> None:
        self.skill_dir = skill_dir
        self.max_retries = max_retries
        self._attempts = 0

    def _missing_artifacts(self) -> list[str]:
        missing = []
        if not (self.skill_dir / "risk_model.pkl").exists():
            missing.append("risk_model.pkl")
        if not (self.skill_dir / "pricing_policy.py").exists():
            missing.append("pricing_policy.py")
        return missing

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        missing = self._missing_artifacts()

        if not missing:
            return None

        if self._attempts >= self.max_retries:
            log.warning(
                "[ResultsGuard] artifacts still missing (%s) after %d retries — giving up",
                ", ".join(missing), self._attempts,
            )
            return None

        self._attempts += 1
        log.warning(
            "[ResultsGuard] missing %s — injecting recovery prompt (attempt %d/%d)",
            ", ".join(missing), self._attempts, self.max_retries,
        )
        return {
            "messages": [
                HumanMessage(
                    content=(
                        f"IMPORTANT: your episode is ending but you have not saved: {', '.join(missing)}.\n\n"
                        "These are your deliverable — the harness independently re-applies them to score "
                        "your episode; it does not use any P&L/C-stat numbers you report yourself. "
                        "Do NOT re-train from scratch. Using the model/pricing logic you already have "
                        "in this session:\n\n"
                        "1. Save your trained risk model as risk_model.pkl in your working directory. "
                        "The pickled object must expose predict_default_proba(X: pd.DataFrame) -> np.ndarray "
                        "returning P(default) per row.\n"
                        "2. Save pricing_policy.py in your working directory, defining "
                        "price(X: pd.DataFrame, p_default_hat: np.ndarray, required_rate: np.ndarray) -> np.ndarray "
                        "returning your chosen offered_rate per row (NaN = decline).\n\n"
                        "A simple pipeline that actually saves these two files beats a sophisticated one "
                        "that doesn't. You may optionally also write results.json with \"approach\" and "
                        "\"hypothesis\" text fields — those are for readability only, not scored."
                    )
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)
