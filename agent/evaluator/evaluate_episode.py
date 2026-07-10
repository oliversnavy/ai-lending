"""
Independent evaluator — the sole source of truth for episode metrics.

Agents no longer self-report pnl/c_stat/acceptance_rate/loans_funded/total_principal.
They submit two artifacts (risk_model.pkl, pricing_policy.py); this module reloads
those artifacts, applies them to a fresh read of the data, and computes every metric
itself using the canonical formula in pnl_formula.py. Nothing the agent writes to
results.json (if anything) is authoritative for these fields — only approach/hypothesis
free text is taken from the agent, for readability in the episode index.

The scoring step (agent-authored model + pricing code) runs in a subprocess via
score_runner.py, which only ever sees val_features_only.parquet (event/observed_time
physically absent). This process never unpickles the agent's model directly and never
holds ground truth in the same process as agent code.
"""
from __future__ import annotations
import pathlib
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from . import pnl_formula

PROJECT_ROOT = pathlib.Path("/home/oliversnavy/repos/ai-lending")
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"  # ground truth — harness-only
SENSITIVITY_BAD = PROJECT_ROOT / "data" / "processed" / "sensitivity_model_defaulter.pkl"
SENSITIVITY_GOOD = PROJECT_ROOT / "data" / "processed" / "sensitivity_model_nondefaulter.pkl"
SCORE_RUNNER = pathlib.Path(__file__).parent / "score_runner.py"
SCORING_TIMEOUT_SECONDS = 300


def _run_score_subprocess(skill_dir: pathlib.Path) -> pd.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = pathlib.Path(tmp) / "scored.parquet"
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            # skill_dir on PYTHONPATH so custom classes the agent's model.pkl depends on
            # (defined in the agent's own scripts) can be found again for unpickling —
            # same issue we hit with SensitivityModel's __main__ pickling earlier today.
            # Deliberately NOT including PROJECT_ROOT / LENDING_DATA_DIR here — raises the
            # bar against accidental leakage. Does not stop a pricing function that
            # hardcodes the real val.parquet's absolute path (known residual risk).
            "PYTHONPATH": str(skill_dir),
        }
        try:
            result = subprocess.run(
                [sys.executable, str(SCORE_RUNNER), str(skill_dir), str(output_path)],
                cwd=str(skill_dir),
                capture_output=True,
                text=True,
                timeout=SCORING_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[evaluator] scoring timed out after {SCORING_TIMEOUT_SECONDS}s")
            return None

        if result.returncode != 0:
            print(f"[evaluator] scoring failed:\n{result.stdout}\n{result.stderr}")
            return None

        if not output_path.exists():
            print("[evaluator] scoring subprocess reported success but wrote no output")
            return None

        return pd.read_parquet(output_path)


def evaluate_episode(skill_dir: pathlib.Path) -> dict:
    """
    Returns a dict with pnl, c_stat, acceptance_rate, loans_funded, total_principal.
    All zero/failed if artifacts are missing or scoring fails — mirrors the old
    "no results.json" -> zeroed EpisodeRecord behavior.
    """
    zero_result = {
        "pnl": 0.0, "c_stat": 0.0, "acceptance_rate": 0.0,
        "loans_funded": 0, "total_principal": 0.0,
        "eval_status": "failed",
    }

    scored = _run_score_subprocess(skill_dir)
    if scored is None:
        return zero_result

    val = pd.read_parquet(VAL_PATH)
    if len(val) != len(scored):
        print(f"[evaluator] row count mismatch: val={len(val)} scored={len(scored)}")
        return zero_result

    p_default_hat = scored["p_default_hat"].values
    offered_rate_raw = scored["offered_rate_raw"].values

    c_stat = float(concordance_index(val["observed_time"], -p_default_hat, val["event"]))

    with open(SENSITIVITY_BAD, "rb") as f:
        model_bad = pickle.load(f)
    with open(SENSITIVITY_GOOD, "rb") as f:
        model_good = pickle.load(f)

    metrics = pnl_formula.evaluate(val, p_default_hat, offered_rate_raw, model_bad, model_good)
    metrics["c_stat"] = c_stat
    metrics["eval_status"] = "ok"
    return metrics
