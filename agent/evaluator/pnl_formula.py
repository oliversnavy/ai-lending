"""
Canonical P&L formula — the ONE implementation used to score every episode, every
treatment. Previously each episode reimplemented this formula from scratch in its own
generated script; that meant 20+ independent chances for the exact class of bug that
produced episode_0005's fabricated result (a pandas indexing mistake), plus no guarantee
different episodes even implemented "the same" formula consistently. This module is
harness-owned and never touched by agent-authored code.

Two phases, deliberately kept separate:
  - DECISION (required_rate, decline, cap ranking): uses only p_default_hat (the agent's
    own ex-ante prediction) and features — never event/observed_time.
  - EVALUATION (realized pnl/loss/interest): uses ground-truth event/observed_time on the
    already-selected portfolio — legitimate, since this is a backtest of a decision already
    made, not a decision itself.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COST_OF_CAPITAL = 0.16
SERVICING_MARGIN = 0.03
AVG_TERM_YEARS = 3.85
LGD_BY_GRADE = {"A": 0.51, "B": 0.55, "C": 0.60, "D": 0.65, "E": 0.70, "F": 0.74, "G": 0.77}
RATE_CEILING = 0.36
CAPITAL_CAP = 2_500_000_000


def compute_required_rate(p_default_hat: np.ndarray, grade: pd.Series) -> np.ndarray:
    lgd = grade.map(LGD_BY_GRADE).values
    p = np.clip(p_default_hat, 0.0, 0.999999)  # guard divide-by-zero at p=1
    risk_margin = (p * lgd) / ((1 - p) * AVG_TERM_YEARS)
    return COST_OF_CAPITAL + SERVICING_MARGIN + risk_margin


def enforce_pricing_invariants(
    offered_rate: np.ndarray, required_rate: np.ndarray
) -> np.ndarray:
    """
    Hard, harness-enforced guarantees that hold for every episode regardless of what
    the agent's pricing_policy.py returned:
      - required_rate > RATE_CEILING  -> forced decline (cannot be priced profitably
        within the legal ceiling, no matter what the agent's function said)
      - otherwise, offered_rate is clipped to [required_rate, RATE_CEILING] (never
        below breakeven, never above the legal ceiling)
      - NaN from the agent's function always means "no offer" and stays NaN
    """
    offered_rate = np.asarray(offered_rate, dtype=float)
    declined_by_economics = required_rate > RATE_CEILING
    result = np.clip(offered_rate, required_rate, RATE_CEILING)
    result[np.isnan(offered_rate)] = np.nan
    result[declined_by_economics] = np.nan
    return result


def rank_and_select_under_cap(
    val: pd.DataFrame,
    offered_rate: np.ndarray,
    p_default_hat: np.ndarray,
    required_rate: np.ndarray,
    p_accept_bad: np.ndarray,
    p_accept_good: np.ndarray,
    cap: float = CAPITAL_CAP,
) -> np.ndarray:
    """
    DECISION phase: which applicants get funded under the capital cap. Uses only
    ex-ante information (p_default_hat, not realized event) — never sees ground truth.
    Returns a boolean mask over val's rows.
    """
    live = ~np.isnan(offered_rate)
    if not live.any():
        return np.zeros(len(val), dtype=bool)

    p_accept_hat = p_default_hat * p_accept_bad + (1 - p_default_hat) * p_accept_good
    est_margin = offered_rate - required_rate
    term_months = val["term"].str.extract(r"(\d+)")[0].astype(float).values
    loan_amnt = val["loan_amnt"].values

    est_pnl_hat = p_accept_hat * est_margin * loan_amnt * (term_months / 12)
    est_principal = p_accept_hat * loan_amnt
    ratio_hat = np.where(est_principal > 0, est_pnl_hat / np.maximum(est_principal, 1e-9), -np.inf)
    ratio_hat = np.where(live, ratio_hat, -np.inf)

    order = np.argsort(-ratio_hat)
    cum_principal = np.cumsum(np.where(live[order], est_principal[order], 0.0))
    cut = np.searchsorted(cum_principal, cap)

    selected_idx = order[:cut]
    mask = np.zeros(len(val), dtype=bool)
    mask[selected_idx] = True
    return mask & live


def realized_pnl(
    val_selected: pd.DataFrame,
    offered_rate_selected: np.ndarray,
    p_accept_bad_selected: np.ndarray,
    p_accept_good_selected: np.ndarray,
) -> dict:
    """
    EVALUATION phase: realized outcomes for the already-selected portfolio, using
    ground-truth event/observed_time. Legitimate here — the selection is already fixed,
    this is scoring a backtest, not making a decision.
    """
    event = val_selected["event"].values
    p_accept = np.where(event == 1, p_accept_bad_selected, p_accept_good_selected)

    r = offered_rate_selected / 12
    term_months = val_selected["term"].str.extract(r"(\d+)")[0].astype(float).values
    t = val_selected["observed_time"].values
    loan_amnt = val_selected["loan_amnt"].values

    monthly_pmt = loan_amnt * r / (1 - (1 + r) ** (-term_months))
    balance_at_t = loan_amnt * ((1 + r) ** term_months - (1 + r) ** t) / ((1 + r) ** term_months - 1)
    interest_per_loan = t * monthly_pmt - (loan_amnt - balance_at_t)
    loss_per_loan = event * p_accept_bad_selected * balance_at_t
    pnl_per_loan = p_accept * interest_per_loan - loss_per_loan

    expected_principal = p_accept * loan_amnt
    return {
        "pnl": float(pnl_per_loan.sum()),
        "total_principal": float(expected_principal.sum()),
        "loans_funded": float(p_accept.sum()),
        "acceptance_rate": float(p_accept.mean()) if len(p_accept) else 0.0,
    }


def evaluate(
    val: pd.DataFrame,
    p_default_hat: np.ndarray,
    offered_rate_raw: np.ndarray,
    sensitivity_model_bad,
    sensitivity_model_good,
) -> dict:
    """
    Full pipeline: raw agent outputs (p_default_hat, offered_rate_raw) -> final metrics.
    `val` must contain ground truth (event, observed_time) — this function is harness-only.
    """
    required_rate = compute_required_rate(p_default_hat, val["grade"])
    offered_rate = enforce_pricing_invariants(offered_rate_raw, required_rate)

    df = val[["grade", "loan_amnt", "annual_inc", "funded_amnt"]].copy()
    df["offered_rate"] = np.where(np.isnan(offered_rate), 0.21, offered_rate)  # placeholder rate for declined rows, filtered out below
    p_accept_bad = sensitivity_model_bad.predict_proba_batch(df)
    p_accept_good = sensitivity_model_good.predict_proba_batch(df)

    mask = rank_and_select_under_cap(
        val, offered_rate, p_default_hat, required_rate, p_accept_bad, p_accept_good
    )

    if not mask.any():
        return {"pnl": 0.0, "c_stat": 0.0, "acceptance_rate": 0.0, "loans_funded": 0, "total_principal": 0.0}

    metrics = realized_pnl(
        val.loc[mask],
        offered_rate[mask],
        p_accept_bad[mask],
        p_accept_good[mask],
    )
    metrics["loans_funded"] = int(round(metrics["loans_funded"]))
    return metrics
