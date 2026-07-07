from __future__ import annotations
import json
import pathlib

import yaml

from agent.app.agent import run_episode
from agent.app.models.episode import EpisodeRecord, LoopMode, TreatmentConfig

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending")


def _load_loop_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)["loop"]


def _check_plateau(
    episode_index: list[EpisodeRecord],
    window: int,
    threshold: float,
) -> bool:
    if len(episode_index) < window + 1:
        return False
    recent = [ep.pnl for ep in episode_index[-(window + 1):]]
    improvements = [recent[i + 1] - recent[i] for i in range(window)]
    return all(abs(imp) / max(abs(recent[-1]), 1) < threshold for imp in improvements)


def run_loop(treatment_config: TreatmentConfig) -> list[EpisodeRecord]:
    """Run the Ralph Wiggum outer loop for a given treatment."""
    cfg = _load_loop_config()
    max_iterations: int = cfg["max_iterations"]
    plateau_window: int = cfg["plateau_window"]
    plateau_threshold: float = cfg["plateau_threshold"]

    episode_index: list[EpisodeRecord] = []
    index_path = (
        PROJECT_ROOT / "results" / "episode_indexes"
        / f"treatment_{treatment_config.treatment}.jsonl"
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(max_iterations):
        print(f"\n[Loop] Treatment {treatment_config.treatment} | Iteration {iteration + 1}/{max_iterations}")

        # Inject based on loop mode
        if treatment_config.loop_mode == LoopMode.SINGLE_PRIOR:
            injected = episode_index[-1:] if episode_index else []
        elif treatment_config.loop_mode == LoopMode.ALL_PRIOR:
            injected = episode_index
        else:
            # No loop — run exactly once
            injected = []

        record = run_episode(treatment_config, injected, len(episode_index))
        episode_index.append(record)

        # Persist to JSONL
        with open(index_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

        print(f"[Loop] P&L: ${record.pnl / 1000:.1f}k | C-stat: {record.c_stat:.3f} | "
              f"Acceptance: {record.acceptance_rate:.1%} | Tokens: {record.tokens_used:,}")

        # Single-episode treatments always stop after one run
        if treatment_config.loop_mode == LoopMode.NONE:
            break

        if _check_plateau(episode_index, plateau_window, plateau_threshold):
            print(f"[Loop] Plateau detected after {len(episode_index)} episodes — stopping.")
            break

    return episode_index
