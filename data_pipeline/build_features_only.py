"""
Build features-only copies of val.parquet and holdout.parquet — with `event` and
`observed_time` physically removed — for the independent evaluator harness.

Why this exists: agent-submitted risk_model.pkl / pricing_policy.py are re-applied by
the harness, not by the agent, at evaluation time. That re-application must never have
access to the realized outcome columns, or a model/pricing function could (accidentally
or deliberately) leak ground truth into a decision that's supposed to be ex-ante. Passing
a DataFrame that structurally lacks those columns closes the accidental case completely.
It does not close a *deliberate* bypass (agent code hardcoding the real file's absolute
path and reading it directly) — that requires real OS-level sandboxing (container / mount
namespace / different user), which doesn't exist in this environment yet. See
known_limitations.md.

Usage:
    uv run python data_pipeline/build_features_only.py
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("data/processed")
DROP_COLS = ["event", "observed_time"]


def build(name: str) -> None:
    src = DATA_DIR / f"{name}.parquet"
    dst = DATA_DIR / f"{name}_features_only.parquet"
    df = pd.read_parquet(src)
    features_only = df.drop(columns=DROP_COLS)
    features_only.to_parquet(dst, index=False)
    print(f"{dst}  ({len(features_only):,} rows, {len(features_only.columns)} cols)")


if __name__ == "__main__":
    build("val")
    build("holdout")
    print("Done.")
