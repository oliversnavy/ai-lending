"""
Independent evaluator — the sole source of truth for episode metrics.

Agents no longer self-report pnl/c_stat/acceptance_rate/loans_funded/total_principal.
They submit two artifacts (risk_model.pkl, pricing_policy.py); this module reloads
those artifacts, applies them to a fresh read of the data, and computes every metric
itself using the canonical formula in pnl_formula.py. Nothing the agent writes to
results.json (if anything) is authoritative for these fields — only approach/hypothesis
free text is taken from the agent, for readability in the episode index.

Two datasets are scored against the SAME saved artifacts, no retraining:
  - holdout.parquet  -> the PRIMARY, reported metrics (pnl/c_stat/etc.). The agent never
    touches this during the episode (system prompt: "NEVER touch during experiments"),
    so it's a genuinely unbiased read on how the strategy generalizes.
  - val.parquet      -> DIAGNOSTIC metrics only (pnl_val/c_stat_val/etc.), kept for
    within-loop analysis. The agent's own local grid search is tuned against this exact
    sample during the episode, so it's expected to look somewhat better than holdout —
    RESEARCH_STRATEGY.md already commits to holdout as the reported number precisely
    because repeatedly testing strategies against the same finite val sample is its own
    (non-leakage) source of overfitting, distinct from the ground-truth-leakage problem
    the features-only split addresses.

The scoring step (agent-authored model + pricing code) runs in a subprocess via
score_runner.py, which only ever sees a features-only file (event/observed_time
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
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SENSITIVITY_BAD = DATA_DIR / "sensitivity_model_defaulter.pkl"
SENSITIVITY_GOOD = DATA_DIR / "sensitivity_model_nondefaulter.pkl"
SCORE_RUNNER = pathlib.Path(__file__).parent / "score_runner.py"
SCORING_TIMEOUT_SECONDS = 300

DATASETS = {
    "holdout": (DATA_DIR / "holdout_features_only.parquet", DATA_DIR / "holdout.parquet"),
    "val": (DATA_DIR / "val_features_only.parquet", DATA_DIR / "val.parquet"),
}

_ZERO_METRICS = {"pnl": 0.0, "c_stat": 0.0, "acceptance_rate": 0.0, "loans_funded": 0, "total_principal": 0.0}


def _run_score_subprocess(skill_dir: pathlib.Path, features_path: pathlib.Path) -> pd.DataFrame | None:
    with tempfile.TemporaryDirectory() as tmp:
        output_path = pathlib.Path(tmp) / "scored.parquet"
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            # skill_dir on PYTHONPATH so custom classes the agent's model.pkl depends on
            # (defined in the agent's own scripts) can be found again for unpickling —
            # same issue we hit with SensitivityModel's __main__ pickling earlier today.
            # Deliberately NOT including PROJECT_ROOT / LENDING_DATA_DIR here — raises the
            # bar against accidental leakage. Does not stop a pricing function that
            # hardcodes the real val.parquet/holdout.parquet's absolute path (known
            # residual risk — see known_limitations.md).
            "PYTHONPATH": str(skill_dir),
        }
        try:
            result = subprocess.run(
                [sys.executable, str(SCORE_RUNNER), str(skill_dir), str(features_path), str(output_path)],
                cwd=str(skill_dir),
                capture_output=True,
                text=True,
                timeout=SCORING_TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[evaluator] scoring timed out after {SCORING_TIMEOUT_SECONDS}s ({features_path.name})")
            return None

        if result.returncode != 0:
            print(f"[evaluator] scoring failed ({features_path.name}):\n{result.stdout}\n{result.stderr}")
            return None

        if not output_path.exists():
            print(f"[evaluator] scoring subprocess reported success but wrote no output ({features_path.name})")
            return None

        return pd.read_parquet(output_path)


def _score_dataset(skill_dir: pathlib.Path, features_path: pathlib.Path, ground_truth_path: pathlib.Path) -> dict | None:
    scored = _run_score_subprocess(skill_dir, features_path)
    if scored is None:
        return None

    truth = pd.read_parquet(ground_truth_path)
    if len(truth) != len(scored):
        print(f"[evaluator] row count mismatch: {ground_truth_path.name}={len(truth)} scored={len(scored)}")
        return None

    p_default_hat = scored["p_default_hat"].values
    offered_rate_raw = scored["offered_rate_raw"].values
    c_stat = float(concordance_index(truth["observed_time"], -p_default_hat, truth["event"]))

    with open(SENSITIVITY_BAD, "rb") as f:
        model_bad = pickle.load(f)
    with open(SENSITIVITY_GOOD, "rb") as f:
        model_good = pickle.load(f)

    metrics = pnl_formula.evaluate(truth, p_default_hat, offered_rate_raw, model_bad, model_good)
    metrics["c_stat"] = c_stat
    return metrics


def evaluate_episode(skill_dir: pathlib.Path) -> dict:
    """
    Returns primary (holdout) metrics as pnl/c_stat/acceptance_rate/loans_funded/
    total_principal, plus val-diagnostic metrics as pnl_val/c_stat_val/etc.
    eval_status is "ok" (both scored), "partial" (only one scored), or "failed" (neither).
    """
    holdout_metrics = _score_dataset(skill_dir, *DATASETS["holdout"])
    val_metrics = _score_dataset(skill_dir, *DATASETS["val"])

    if holdout_metrics is None and val_metrics is None:
        status = "failed"
    elif holdout_metrics is None or val_metrics is None:
        status = "partial"
    else:
        status = "ok"

    result = dict(holdout_metrics or _ZERO_METRICS)
    result.update({f"{k}_val": v for k, v in (val_metrics or _ZERO_METRICS).items()})
    result["eval_status"] = status
    return result
