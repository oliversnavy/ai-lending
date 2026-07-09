"""
Synthetic customer acceptance model for the ai-lending simulation.

Models the probability that a borrower accepts a loan offer, given their
applicant features and the proposed offer terms.

Why synthetic (not data-trained): the LendingClub dataset contains only
funded loans — applications that LendingClub approved AND the borrower
accepted. No borrower-rejection signal exists in the data. A synthetic
model with calibrated parameters avoids this selection bias and provides
a known ground truth that can be verified analytically.

The model reflects a non-bank fintech lender with a high cost of capital.
For prime borrowers (Grade A/B), the minimum viable rate exceeds market
rates, so acceptance probability collapses regardless of pricing.
The viable zone is approximately Grade C–F.

Two-model adverse selection design
-----------------------------------
The simulation ships two sensitivity models with different rate elasticities:

  sensitivity_model_nondefaulter.pkl  beta_spread=16.0
    Non-defaulters have outside credit options. When offered above-market
    rates they walk away — high rate sensitivity.

  sensitivity_model_defaulter.pkl     beta_spread=8.0
    Defaulters are credit-constrained. They accept even at above-market
    rates — low rate sensitivity.

Together these create realistic adverse selection: charging above-market
rates increasingly fills the accepted pool with higher-risk borrowers,
raising expected loss faster than interest income. Flat 36% to Grade C/D
is therefore suboptimal — the effective default rate in the accepted pool
rises by ~15-23 pp, swamping the rate premium.

Usage:
    uv run python data_pipeline/sensitivity_model.py  # calibrate and save both models
    from data_pipeline.sensitivity_model import SensitivityModel, load  # load and use
"""

import pathlib
import pickle
import numpy as np
import pandas as pd


OUT_PATH            = pathlib.Path("data/processed/sensitivity_model.pkl")
OUT_PATH_DEFAULTER  = pathlib.Path("data/processed/sensitivity_model_defaulter.pkl")
OUT_PATH_NONDEFAULTER = pathlib.Path("data/processed/sensitivity_model_nondefaulter.pkl")

# ---------------------------------------------------------------------------
# Cost of capital / minimum viable rate
# ---------------------------------------------------------------------------

COST_OF_CAPITAL = 0.16       # blended warehouse + ABS funding cost
SERVICING_MARGIN = 0.03      # minimum margin to cover servicing costs
EXPECTED_LOSS_BUFFER = 0.02  # baseline loss provision added to floor

MIN_VIABLE_RATE = COST_OF_CAPITAL + SERVICING_MARGIN + EXPECTED_LOSS_BUFFER  # 0.21

# ---------------------------------------------------------------------------
# Market rates by grade — what the borrower can get elsewhere.
# Calibrated to approximate real consumer lending tier pricing.
# Agent is NOT given these directly; it must infer the competitive
# landscape from acceptance feedback across episodes.
# ---------------------------------------------------------------------------

MARKET_RATE_BY_GRADE = {
    "A": 0.095,   # 9.5%  — prime, many bank alternatives
    "B": 0.140,   # 14.0% — near-prime, some bank alternatives
    "C": 0.190,   # 19.0% — borderline, limited alternatives
    "D": 0.240,   # 24.0% — subprime, few alternatives
    "E": 0.295,   # 29.5% — deep subprime, very few alternatives
    "F": 0.340,   # 34.0% — high risk, almost no alternatives
    "G": 0.370,   # 37.0% — very high risk
}

# ---------------------------------------------------------------------------
# Logistic acceptance model
#
# P(accept) = sigmoid(
#     α₀
#     - β_spread  * max(0, offered_rate - market_rate)  # rate competitiveness
#     - β_burden  * (loan_amnt / annual_inc)             # payment burden
#     + β_match   * (loan_amnt / funded_amnt)            # fulfillment ratio
#     + ε                                                # noise
# )
#
# Coefficients calibrated so that:
#   - A borrower offered exactly at market rate has ~55% acceptance
#   - Each 5pp above market rate reduces acceptance by ~15-20pp
#   - Grade A/B borrowers offered at MIN_VIABLE_RATE accept at <5%
#   - Grade D/E borrowers offered at MIN_VIABLE_RATE accept at ~40-50%
# ---------------------------------------------------------------------------

ALPHA_0 = 0.20      # base log-odds (~55% at market rate, neutral burden)
BETA_SPREAD = 12.0  # legacy single-model value (kept for reference)
BETA_SPREAD_NONDEFAULTER = 16.0  # non-defaulters: high rate sensitivity (have outside options)
BETA_SPREAD_DEFAULTER    =  8.0  # defaulters: low rate sensitivity (credit-constrained)
BETA_BURDEN = 1.5  # sensitivity to loan amount / income ratio
BETA_MATCH = 0.50  # reward for funding close to what borrower requested
NOISE_STD = 0.30   # stochastic noise (ε ~ N(0, NOISE_STD))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SensitivityModel:
    __module__ = "data_pipeline.sensitivity_model"  # ensures correct pickle path when run as __main__
    """
    Callable acceptance model. Given applicant features and offer terms,
    returns P(accept). Held fixed across all 7 experimental treatments.

    Two pre-built variants are saved to data/processed/:
      sensitivity_model_nondefaulter.pkl  (beta_spread=16.0 — rate-sensitive)
      sensitivity_model_defaulter.pkl     (beta_spread=8.0  — rate-inelastic)

    Use both together to model adverse selection; see the P&L formula in the
    system prompt. Which curve applies to a given historical loan is selected
    by the realized `event` outcome (ground truth), not by the risk model's
    p_default_hat — this is a backtest, so the borrower's true type is known.
    """

    def __init__(
        self,
        market_rates: dict = MARKET_RATE_BY_GRADE,
        min_viable_rate: float = MIN_VIABLE_RATE,
        alpha_0: float = ALPHA_0,
        beta_spread: float = BETA_SPREAD,
        beta_burden: float = BETA_BURDEN,
        beta_match: float = BETA_MATCH,
        noise_std: float = NOISE_STD,
        random_seed: int = 42,
    ):
        self.market_rates = market_rates
        self.min_viable_rate = min_viable_rate
        self.alpha_0 = alpha_0
        self.beta_spread = beta_spread
        self.beta_burden = beta_burden
        self.beta_match = beta_match
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_seed)

    def predict_proba(
        self,
        grade: str,
        offered_rate: float,
        loan_amnt: float,
        annual_inc: float,
        funded_amnt: float,
    ) -> float:
        """
        Returns P(accept) for a single applicant + offer.

        Args:
            grade:        LendingClub grade (A–G)
            offered_rate: annual interest rate offered (0.0–1.0)
            loan_amnt:    dollar amount offered
            annual_inc:   borrower annual income
            funded_amnt:  amount originally requested (from dataset)

        Returns:
            float in [0, 1]
        """
        # Offer below minimum viable rate is not allowed
        if offered_rate < self.min_viable_rate:
            return 0.0

        market_rate = self.market_rates.get(grade, 0.25)
        rate_spread = max(0.0, offered_rate - market_rate)
        burden = loan_amnt / max(annual_inc, 1.0)
        match = min(loan_amnt / max(funded_amnt, 1.0), 1.0)
        noise = self.rng.normal(0, self.noise_std)

        log_odds = (
            self.alpha_0
            - self.beta_spread * rate_spread
            - self.beta_burden * burden
            + self.beta_match * match
            + noise
        )
        return float(_sigmoid(np.array(log_odds)))

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized batch prediction. df must have columns:
        grade, offered_rate, loan_amnt, annual_inc, funded_amnt.
        """
        market_rates = df["grade"].map(self.market_rates).fillna(0.25).values
        rate_spread = np.maximum(0.0, df["offered_rate"].values - market_rates)
        burden = df["loan_amnt"].values / np.maximum(df["annual_inc"].values, 1.0)
        match = np.minimum(
            df["loan_amnt"].values / np.maximum(df["funded_amnt"].values, 1.0), 1.0
        )
        noise = self.rng.normal(0, self.noise_std, size=len(df))

        log_odds = (
            self.alpha_0
            - self.beta_spread * rate_spread
            - self.beta_burden * burden
            + self.beta_match * match
            + noise
        )
        # Zero out offers below minimum viable rate
        below_floor = df["offered_rate"].values < self.min_viable_rate
        probs = _sigmoid(log_odds)
        probs[below_floor] = 0.0
        return probs


def calibrate_and_show() -> tuple["SensitivityModel", "SensitivityModel"]:
    """Build both sensitivity models and print a calibration comparison."""
    model_good = SensitivityModel(beta_spread=BETA_SPREAD_NONDEFAULTER)
    model_bad  = SensitivityModel(beta_spread=BETA_SPREAD_DEFAULTER)

    DEFAULT_RATES = {"C": 0.180, "D": 0.239, "E": 0.306, "F": 0.352}

    def _mean_p(model, grade, rate, n=500):
        return np.mean([model.predict_proba(grade, rate, 15000, 65000, 15000) for _ in range(n)])

    print("Adverse selection check — effective default rate in accepted pool")
    print(f"{'Grade':6s} {'Rate':>6s}  {'p_acc(good)':>11s}  {'p_acc(bad)':>10s}  {'p_acc_eff':>10s}  {'eff_def_rate':>13s}  {'vs_base':>8s}")
    print("-" * 75)
    for grade in ["C", "D", "E", "F"]:
        pd_ = DEFAULT_RATES[grade]
        mkt = MARKET_RATE_BY_GRADE[grade]
        for rate in [mkt, 0.28, 0.36]:
            pg = _mean_p(model_good, grade, rate)
            pb = _mean_p(model_bad,  grade, rate)
            pe = pd_ * pb + (1 - pd_) * pg
            eff_def = (pd_ * pb / pe) if pe > 0 else 0
            print(f"{grade:6s} {rate:>6.1%}  {pg:>11.1%}  {pb:>10.1%}  {pe:>10.1%}  {eff_def:>13.1%}  {eff_def-pd_:>+8.1%}")
        print()

    return model_good, model_bad


def save(model: "SensitivityModel", path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved → {path}")


def load(path: pathlib.Path) -> "SensitivityModel":
    with open(path, "rb") as f:
        return pickle.load(f)


def load_nondefaulter(path: pathlib.Path = OUT_PATH_NONDEFAULTER) -> "SensitivityModel":
    return load(path)


def load_defaulter(path: pathlib.Path = OUT_PATH_DEFAULTER) -> "SensitivityModel":
    return load(path)


if __name__ == "__main__":
    # Import via importlib so SensitivityModel is pickled as data_pipeline.sensitivity_model.SensitivityModel
    # (not __main__.SensitivityModel), which allows loading from any other module.
    import importlib
    _mod = importlib.import_module("data_pipeline.sensitivity_model")
    _good = _mod.SensitivityModel(beta_spread=_mod.BETA_SPREAD_NONDEFAULTER)
    _bad  = _mod.SensitivityModel(beta_spread=_mod.BETA_SPREAD_DEFAULTER)
    calibrate_and_show()
    _mod.save(_good, _mod.OUT_PATH_NONDEFAULTER)
    _mod.save(_bad,  _mod.OUT_PATH_DEFAULTER)
