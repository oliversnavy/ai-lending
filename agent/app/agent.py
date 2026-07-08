from __future__ import annotations
import json
import logging
import pathlib
import time

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .middleware.episode_injector import build_initial_human_message
from .models.episode import EpisodeRecord, TreatmentConfig
from ..graph import build_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

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
    final_state: dict = {"messages": []}
    log.info("[T%s ep%04d] Starting episode", treatment_config.treatment, episode_id)

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=initial_message)]},
        config=config,
        stream_mode="updates",
    ):
        for node, delta in chunk.items():
            if not delta:
                continue
            for msg in delta.get("messages", []):
                if isinstance(msg, AIMessage):
                    calls = getattr(msg, "tool_calls", [])
                    if calls:
                        for tc in calls:
                            args_preview = str(tc.get("args", {}))[:120]
                            log.info("  [model→tool] %s(%s)", tc["name"], args_preview)
                    elif msg.content:
                        preview = str(msg.content)[:200].replace("\n", " ")
                        log.info("  [model→done] %s", preview)
                elif isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == "summarization":
                    log.info("  [summarize ] context compressed — history trimmed")
                elif isinstance(msg, ToolMessage):
                    result_preview = str(msg.content)[:120].replace("\n", " ")
                    log.info("  [tool→model] %s → %s", msg.name, result_preview)
            # accumulate final state
            if "messages" in delta:
                final_state["messages"] = final_state["messages"] + delta["messages"]

    duration_s = time.time() - t0
    log.info("[T%s ep%04d] Done in %.1fs", treatment_config.treatment, episode_id, duration_s)

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
