"""
Runs INSIDE a subprocess, isolated from the harness's main process. This is the only
place agent-authored code (risk_model.pkl's unpickled class, pricing_policy.py) ever
executes. It is only ever given val_features_only.parquet — event/observed_time are
physically absent from that file, so there is nothing to accidentally leak. (A pricing
function that deliberately hardcodes the real val.parquet's absolute path and reads it
directly is a known, unclosed residual risk — see known_limitations.md; closing that
needs real OS-level sandboxing this environment doesn't have yet.)

Usage (invoked by evaluate_episode.py, not run directly):
    python score_runner.py <skill_dir> <output_path>

Writes a parquet with columns: p_default_hat, offered_rate_raw, required_rate
"""
from __future__ import annotations
import importlib.util
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

FEATURES_ONLY_PATH = pathlib.Path("/home/oliversnavy/repos/ai-lending/data/processed/val_features_only.parquet")


def compute_required_rate(p_default_hat: np.ndarray, grade: pd.Series) -> np.ndarray:
    COST_OF_CAPITAL, SERVICING_MARGIN, AVG_TERM_YEARS = 0.16, 0.03, 3.85
    LGD_BY_GRADE = {"A": 0.51, "B": 0.55, "C": 0.60, "D": 0.65, "E": 0.70, "F": 0.74, "G": 0.77}
    lgd = grade.map(LGD_BY_GRADE).values
    p = np.clip(p_default_hat, 0.0, 0.999999)
    risk_margin = (p * lgd) / ((1 - p) * AVG_TERM_YEARS)
    return COST_OF_CAPITAL + SERVICING_MARGIN + risk_margin


def load_pricing_policy(skill_dir: pathlib.Path):
    policy_path = skill_dir / "pricing_policy.py"
    if not policy_path.exists():
        raise FileNotFoundError(f"pricing_policy.py not found in {skill_dir}")
    spec = importlib.util.spec_from_file_location("pricing_policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "price"):
        raise AttributeError("pricing_policy.py must define a `price(X, p_default_hat, required_rate)` function")
    return module.price


def main() -> None:
    skill_dir = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])

    model_path = skill_dir / "risk_model.pkl"
    if not model_path.exists():
        print(f"ERROR: risk_model.pkl not found in {skill_dir}", file=sys.stderr)
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_default_proba"):
        print("ERROR: risk_model.pkl must expose a predict_default_proba(X) method", file=sys.stderr)
        sys.exit(1)

    X = pd.read_parquet(FEATURES_ONLY_PATH)  # features only — no event/observed_time present in this file

    p_default_hat = np.asarray(model.predict_default_proba(X), dtype=float)
    if p_default_hat.shape[0] != len(X):
        print(f"ERROR: predict_default_proba returned {p_default_hat.shape[0]} rows, expected {len(X)}", file=sys.stderr)
        sys.exit(1)

    required_rate = compute_required_rate(p_default_hat, X["grade"])

    price_fn = load_pricing_policy(skill_dir)
    offered_rate_raw = np.asarray(price_fn(X, p_default_hat, required_rate), dtype=float)
    if offered_rate_raw.shape[0] != len(X):
        print(f"ERROR: price() returned {offered_rate_raw.shape[0]} rows, expected {len(X)}", file=sys.stderr)
        sys.exit(1)

    out = pd.DataFrame({
        "p_default_hat": p_default_hat,
        "offered_rate_raw": offered_rate_raw,
        "required_rate": required_rate,
    })
    out.to_parquet(output_path, index=False)
    print(f"Scored {len(out):,} rows -> {output_path}")


if __name__ == "__main__":
    main()
