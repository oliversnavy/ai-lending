"""
Trace and episode analysis utilities for the AI lending ablation study.

Reusable across all treatments — pass treatment ID (e.g. "0", "1a", "1b")
to get summaries, trace breakdowns, model detection, and power analysis.

Usage:
    from scripts.trace_analysis import load_episodes, load_trace, treatment_summary
"""
from __future__ import annotations

import json
import re
import pathlib
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
TRACES_DIR   = RESULTS_DIR / "traces"
INDEX_DIR    = RESULTS_DIR / "episode_indexes"
SKILLS_DIR   = RESULTS_DIR / "skills"


# ---------------------------------------------------------------------------
# Episode index
# ---------------------------------------------------------------------------

def load_episodes(treatment: str) -> pd.DataFrame:
    """Load all episode records for a treatment from the JSONL index."""
    path = INDEX_DIR / f"treatment_{treatment}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No episode index for treatment {treatment}: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records).sort_values("episode_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_trace(treatment: str, episode_id: int) -> list[dict]:
    """Load the full JSONL trace for one episode. Returns [] if no trace exists."""
    path = TRACES_DIR / f"treatment_{treatment}" / f"episode_{episode_id:04d}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def has_trace(treatment: str, episode_id: int) -> bool:
    return (TRACES_DIR / f"treatment_{treatment}" / f"episode_{episode_id:04d}.jsonl").exists()


# ---------------------------------------------------------------------------
# Trace feature extraction
# ---------------------------------------------------------------------------

_MODEL_PATTERNS: list[tuple[str, str]] = [
    (r"\blightgbm\b|\bLGBMClassifier\b|\bLGBMModel\b",    "lightgbm"),
    (r"\bxgboost\b|\bXGBClassifier\b|\bXGBModel\b",       "xgboost"),
    (r"\bcatboost\b|\bCatBoostClassifier\b",               "catboost"),
    (r"\bDeepHit\b",                                        "deephit"),
    (r"\bCoxPH\b|\bDeepSurv\b",                            "coxph_nn"),
    (r"\bPCHazard\b",                                       "pchazard"),
    (r"\bRandomForest\b|\bRandomForestClassifier\b",        "random_forest"),
    (r"\bGradientBoosting\b|\bHistGradientBoosting\b",     "sklearn_gbm"),
    (r"\bLogisticRegression\b",                            "logistic"),
    (r"\bCoxPHFitter\b",                                   "lifelines_cox"),
    (r"\bSurvivalAnalysis\b|\bWeibull\b",                  "survival_other"),
]

def detect_models(trace: list[dict]) -> list[str]:
    """Return sorted list of unique model types used, detected from tool outputs."""
    found: set[str] = set()
    for msg in trace:
        if msg.get("type") != "ToolMessage":
            continue
        content = str(msg.get("content", ""))
        for pattern, label in _MODEL_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                found.add(label)
    # Also check AI message tool_call args (what was submitted, not just stdout)
    for msg in trace:
        if msg.get("type") != "AIMessage":
            continue
        for tc in msg.get("tool_calls", []):
            code = str(tc.get("args", {}).get("code", ""))
            for pattern, label in _MODEL_PATTERNS:
                if re.search(pattern, code, re.IGNORECASE):
                    found.add(label)
    return sorted(found)


def count_tool_calls(trace: list[dict]) -> int:
    return sum(1 for m in trace if m.get("type") == "ToolMessage")


def count_errors(trace: list[dict]) -> dict[str, int]:
    """Count stderr errors and timeouts in tool results."""
    errors, timeouts = 0, 0
    for msg in trace:
        if msg.get("type") != "ToolMessage":
            continue
        content = str(msg.get("content", ""))
        if "Traceback" in content or "Error" in content or "STDERR" in content:
            errors += 1
        if "timed out" in content.lower() or "timeout" in content.lower():
            timeouts += 1
    return {"errors": errors, "timeouts": timeouts}


def count_summarizations(trace: list[dict]) -> int:
    """Count how many times SummarizationMiddleware compressed context."""
    return sum(
        1 for m in trace
        if m.get("type") == "HumanMessage" and m.get("source") == "summarization"
    )


def count_overflow_recoveries(trace: list[dict]) -> int:
    """Count OverflowRecoveryMiddleware retries (look for [overflow-cleared] in tool results)."""
    return sum(
        1 for m in trace
        if m.get("type") == "ToolMessage" and "[overflow-cleared]" in str(m.get("content", ""))
    )


def extract_thinking(trace: list[dict]) -> list[str]:
    """Return all thinking block texts from the trace (AI internal reasoning)."""
    return [m["thinking"] for m in trace if m.get("type") == "AIMessage" and m.get("thinking")]


def total_tokens(trace: list[dict]) -> int:
    """Sum input tokens across all AI turns (approximates total usage)."""
    return sum(
        m.get("usage", {}).get("total_tokens", 0)
        for m in trace if m.get("type") == "AIMessage"
    )


def episode_trace_features(treatment: str, episode_id: int) -> dict[str, Any]:
    """Return a feature dict for one episode combining results.json + trace analysis."""
    trace = load_trace(treatment, episode_id)
    skill_dir = SKILLS_DIR / f"treatment_{treatment}" / f"episode_{episode_id:04d}"
    results_path = skill_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    return {
        "episode_id":          episode_id,
        "has_trace":           len(trace) > 0,
        "pnl":                 results.get("pnl", 0),
        "c_stat":              results.get("c_stat", 0),
        "acceptance_rate":     results.get("acceptance_rate", 0),
        "loans_funded":        results.get("loans_funded", 0),
        "approach":            results.get("approach", ""),
        "hypothesis":          results.get("hypothesis", ""),
        "models_used":         detect_models(trace),
        "tool_calls":          count_tool_calls(trace),
        "errors":              count_errors(trace)["errors"],
        "timeouts":            count_errors(trace)["timeouts"],
        "summarizations":      count_summarizations(trace),
        "overflow_recoveries": count_overflow_recoveries(trace),
        "thinking_turns":      len(extract_thinking(trace)),
        "trace_tokens":        total_tokens(trace),
    }


def treatment_trace_df(treatment: str, episode_ids: list[int] | None = None) -> pd.DataFrame:
    """Build a per-episode feature DataFrame for all traced episodes in a treatment."""
    episodes = load_episodes(treatment)
    if episode_ids is None:
        episode_ids = episodes["episode_id"].tolist()
    rows = [episode_trace_features(treatment, eid) for eid in episode_ids]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float] | np.ndarray,
    stat: str = "mean",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    fn = np.mean if stat == "mean" else np.median
    point = float(fn(arr))
    boots = [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def required_n(
    sd: float,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Estimate sample size needed per group to detect `effect_size` given pooled `sd`."""
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * sd / effect_size) ** 2
    return int(np.ceil(n))


def summary_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    vals = df[col].dropna()
    mean, lo, hi = bootstrap_ci(vals, stat="mean")
    return {
        "n":      len(vals),
        "mean":   mean,
        "sd":     float(vals.std()),
        "median": float(vals.median()),
        "p10":    float(vals.quantile(0.10)),
        "p90":    float(vals.quantile(0.90)),
        "ci_lo":  lo,
        "ci_hi":  hi,
        "min":    float(vals.min()),
        "max":    float(vals.max()),
    }


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def flag_outliers(df: pd.DataFrame, col: str, z_thresh: float = 2.0) -> pd.Series:
    """Return boolean Series: True where |z-score| > z_thresh."""
    z = (df[col] - df[col].mean()) / df[col].std()
    return z.abs() > z_thresh


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return episodes flagged as outliers on P&L or C-stat."""
    pnl_out  = flag_outliers(df, "pnl")
    cstat_out = flag_outliers(df, "c_stat")
    flagged = df[pnl_out | cstat_out].copy()
    flagged["flag_pnl"]   = pnl_out[flagged.index]
    flagged["flag_cstat"] = cstat_out[flagged.index]
    return flagged[["episode_id", "pnl", "c_stat", "approach", "flag_pnl", "flag_cstat"]]
