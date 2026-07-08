#!/usr/bin/env python
"""
CLI entry point for running an ablation treatment.

Examples:
    uv run python agent/run.py --treatment 0  --n-runs 20   # vanilla ReAct baseline
    uv run python agent/run.py --treatment 1a --n-runs 20   # deep agent 35B-A3B
    uv run python agent/run.py --treatment 3                 # GEPA + single-prior loop
"""
from __future__ import annotations
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import click

from agent.app.models.episode import TREATMENT_CONFIGS
from agent.loop import run_loop

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results" / "skills"


def _next_episode_id(treatment: str) -> int:
    """Return the next episode ID not yet present on disk."""
    treatment_dir = RESULTS_DIR / f"treatment_{treatment}"
    if not treatment_dir.exists():
        return 0
    existing = sorted(
        int(p.name.split("_")[1])
        for p in treatment_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    return existing[-1] + 1 if existing else 0


@click.command()
@click.option("--treatment", required=True,
              type=click.Choice(list(TREATMENT_CONFIGS.keys())),
              help="Treatment ID (0, 1a, 1b, 2, 3, 4, 5, 6, 7)")
@click.option("--n-runs", default=None, type=int,
              help="Episodes to run. No-loop treatments default to 20; "
                   "loop treatments run until plateau.")
def main(treatment: str, n_runs: int | None) -> None:
    config = TREATMENT_CONFIGS[treatment]

    if config.is_baseline:
        n = n_runs or 20
        start_id = _next_episode_id(treatment)
        print(f"Running Treatment {treatment} — {n} independent episodes (ep {start_id}–{start_id+n-1})")
        from agent.app.agent import run_episode
        index_path = RESULTS_DIR.parent / "episode_indexes" / f"treatment_{treatment}.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        all_records = []
        for i in range(n):
            episode_id = start_id + i
            print(f"\n=== Run {i + 1}/{n} (ep {episode_id:04d}) ===")
            record = run_episode(config, [], episode_id)
            all_records.append(record)
            with open(index_path, "a") as f:
                f.write(record.model_dump_json() + "\n")
            print(f"  P&L: ${record.pnl/1000:.1f}k | C-stat: {record.c_stat:.3f} "
                  f"| Tokens: {record.tokens_used:,}")
    else:
        if n_runs:
            print(f"Note: --n-runs={n_runs} noted (loop uses plateau detection from configs/base.yaml)")
        all_records = run_loop(config)

    pnls = [r.pnl for r in all_records]
    if pnls:
        print(f"\n{'='*50}")
        print(f"Treatment {treatment} complete — {len(pnls)} episode(s)")
        if len(pnls) > 1:
            print(f"  P&L  mean: ${statistics.mean(pnls)/1000:.1f}k  "
                  f"stdev: ${statistics.stdev(pnls)/1000:.1f}k")
        else:
            print(f"  P&L: ${pnls[0]/1000:.1f}k")


if __name__ == "__main__":
    main()
