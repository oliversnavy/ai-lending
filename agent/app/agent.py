from __future__ import annotations
import json
import pathlib
import time

from langchain_core.messages import HumanMessage

from .middleware.episode_injector import build_initial_human_message
from .models.episode import EpisodeRecord, TreatmentConfig
from ..graph import build_graph

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending")
RESULTS_DIR = PROJECT_ROOT / "results"


def get_skill_dir(treatment: str, episode_id: int) -> pathlib.Path:
    d = RESULTS_DIR / "skills" / f"treatment_{treatment}" / f"episode_{episode_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_episode(
    treatment_config: TreatmentConfig,
    episode_index: list[EpisodeRecord],
    episode_id: int,
) -> EpisodeRecord:
    """Run one full agent episode and return the logged EpisodeRecord."""
    skill_dir = get_skill_dir(treatment_config.treatment, episode_id)

    initial_message = build_initial_human_message(
        treatment_config, episode_index, str(skill_dir), episode_id
    )

    graph, callbacks = build_graph(treatment_config, skill_dir)

    config = {"callbacks": callbacks} if callbacks else {}
    t0 = time.time()
    final_state = graph.invoke(
        {"messages": [HumanMessage(content=initial_message)]},
        config=config,
    )
    duration_s = time.time() - t0

    tokens_used = _count_tokens(final_state.get("messages", []))

    return _read_results(skill_dir, episode_id, treatment_config.treatment, tokens_used, duration_s)


def _count_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            total += meta.get("total_tokens", 0) or 0
    return total


def _read_results(
    skill_dir: pathlib.Path,
    episode_id: int,
    treatment: str,
    tokens_used: int,
    duration_s: float,
) -> EpisodeRecord:
    results_path = skill_dir / "results.json"

    if results_path.exists():
        data = json.loads(results_path.read_text())
    else:
        data = {
            "pnl": 0.0,
            "c_stat": 0.0,
            "acceptance_rate": 0.0,
            "loans_funded": 0,
            "total_principal": 0.0,
            "approach": "Agent did not complete (no results.json produced).",
            "hypothesis": "N/A",
        }

    return EpisodeRecord(
        episode_id=episode_id,
        treatment=treatment,
        pnl=float(data.get("pnl", 0)),
        c_stat=float(data.get("c_stat", 0)),
        acceptance_rate=float(data.get("acceptance_rate", 0)),
        loans_funded=int(data.get("loans_funded", 0)),
        total_principal=float(data.get("total_principal", 0)),
        approach=str(data.get("approach", "")),
        hypothesis=str(data.get("hypothesis", "")),
        skill_path=str(skill_dir.relative_to(PROJECT_ROOT)),
        tokens_used=tokens_used,
        duration_s=round(duration_s, 1),
    )
