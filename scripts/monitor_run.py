"""
Live monitor for an in-progress treatment run.
Tails the episode index JSONL and prints each new episode as it arrives,
flagging flat-rate strategies and notable P&L outliers.

Usage:
    uv run python scripts/monitor_run.py            # watch treatment 0
    uv run python scripts/monitor_run.py --treatment 1a
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import time

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
INDEX_DIR    = PROJECT_ROOT / "results" / "episode_indexes"

FLAT36_PATTERNS = [
    r"flat 36",
    r"36% to all",
    r"36% apr to all",
    r"36%.*fixed",
    r"fixed.*36%",
    r"all.*36%",
    r"uniform.*36%",
]

def _is_flat36(approach: str) -> bool:
    a = approach.lower()
    return any(re.search(p, a) for p in FLAT36_PATTERNS)

def _format_row(r: dict) -> str:
    ep   = r.get("episode_id", "?")
    pnl  = r.get("pnl", 0)
    c    = r.get("c_stat", 0)
    acc  = r.get("acceptance_rate", 0)
    dur  = r.get("duration_s", 0)
    approach = r.get("approach", "")[:90]
    flag = "  *** FLAT 36% ***" if _is_flat36(approach) else ""
    pnl_flag = "  *** LOW ***" if pnl < 5e6 else ("  *** HIGH ***" if pnl > 22e6 else "")
    return (
        f"ep{ep:04d}  P&L ${pnl/1e6:6.2f}M  C={c:.3f}  acc={acc:.3f}  {dur/60:.1f}min"
        f"{pnl_flag}{flag}\n"
        f"         {approach}"
    )

def tail(treatment: str, poll_interval: float = 20.0) -> None:
    path = INDEX_DIR / f"treatment_{treatment}.jsonl"
    seen: set[int] = set()
    print(f"Watching {path}  (poll every {poll_interval:.0f}s) — Ctrl-C to stop\n")
    print(f"{'ep':6s}  {'P&L':>10s}  {'C-stat':>7s}  {'AccRate':>8s}  {'Time':>6s}  Approach")
    print("-" * 110)

    while True:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = r.get("episode_id", -1)
                    if eid not in seen:
                        seen.add(eid)
                        print(_format_row(r))
                        print()
        time.sleep(poll_interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--treatment", default="0")
    p.add_argument("--poll", type=float, default=20.0)
    args = p.parse_args()
    tail(args.treatment, args.poll)
