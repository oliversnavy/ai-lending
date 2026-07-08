from __future__ import annotations
import json
import logging
import pathlib
import time
from typing import IO

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from .middleware.episode_injector import build_initial_human_message
from .models.episode import EpisodeRecord, TreatmentConfig
from ..graph import build_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending")
RESULTS_DIR = PROJECT_ROOT / "results"
TRACES_DIR = RESULTS_DIR / "traces"


def get_skill_dir(treatment: str, episode_id: int) -> pathlib.Path:
    d = RESULTS_DIR / "skills" / f"treatment_{treatment}" / f"episode_{episode_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serialize_message(msg: AnyMessage, node: str, turn: int) -> dict:
    """Serialise a LangChain message to a trace-friendly dict."""
    entry: dict = {
        "ts": time.strftime("%H:%M:%S"),
        "turn": turn,
        "node": node,
        "type": type(msg).__name__,
    }

    # Content — can be str or list of blocks (thinking model returns list)
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        # Separate thinking blocks from text blocks for readability
        thinking = [b.get("thinking", "") for b in content if isinstance(b, dict) and b.get("type") == "thinking"]
        text = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if thinking:
            entry["thinking"] = "\n".join(thinking)
        entry["content"] = "\n".join(text) if text else content
    else:
        entry["content"] = content

    # Tool calls on AI messages
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = tool_calls

    # Tool message metadata
    name = getattr(msg, "name", None)
    if name:
        entry["name"] = name
    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id

    # Token usage
    usage = getattr(msg, "usage_metadata", None)
    if usage:
        entry["usage"] = usage

    # LangChain source tag (e.g. "summarization")
    lc_source = getattr(msg, "additional_kwargs", {}).get("lc_source")
    if lc_source:
        entry["source"] = lc_source

    return entry


def _write_trace(f: IO[str], msg: AnyMessage, node: str, turn: int) -> None:
    try:
        entry = _serialize_message(msg, node, turn)
        f.write(json.dumps(entry, default=str) + "\n")
        f.flush()
    except Exception as exc:
        log.warning("Trace write failed for message type %s: %s", type(msg).__name__, exc)


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

    trace_path = TRACES_DIR / f"treatment_{treatment_config.treatment}" / f"episode_{episode_id:04d}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    turn = 0

    with open(trace_path, "w") as trace_f:
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=initial_message)]},
                config=config,
                stream_mode="updates",
            ):
                for node, delta in chunk.items():
                    if not delta:
                        continue
                    for msg in delta.get("messages", []):
                        _write_trace(trace_f, msg, node, turn)
                        turn += 1

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

                    if "messages" in delta:
                        final_state["messages"] = final_state["messages"] + delta["messages"]

        except Exception as e:
            log.error("[T%s ep%04d] Episode crashed: %s", treatment_config.treatment, episode_id, e)

    log.info("[T%s ep%04d] Trace written → %s (%d messages)", treatment_config.treatment, episode_id, trace_path, turn)

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
