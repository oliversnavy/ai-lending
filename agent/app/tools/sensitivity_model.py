from __future__ import annotations
import pathlib
import pickle

from langchain_core.tools import tool

_MODEL = None
_MODEL_PATH = pathlib.Path("data/processed/sensitivity_model.pkl")


def _get_model():
    global _MODEL
    if _MODEL is None:
        with open(_MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)
    return _MODEL


@tool
def sensitivity_model_query(
    grade: str,
    offered_rate: float,
    loan_amnt: float,
    annual_inc: float,
    funded_amnt: float,
) -> str:
    """
    Spot-check the probability that a borrower accepts a loan offer.

    For bulk evaluation across many applicants, load the pickle directly in
    code_executor instead:
        import pickle
        model = pickle.load(open('data/processed/sensitivity_model.pkl', 'rb'))
        probs = model.predict_proba_batch(df)  # df needs: grade, offered_rate,
                                               #   loan_amnt, annual_inc, funded_amnt

    Args:
        grade:        LendingClub grade letter (A–G).
        offered_rate: Annual interest rate offered, as a decimal (e.g. 0.24 for 24%).
        loan_amnt:    Dollar amount offered.
        annual_inc:   Borrower annual income in dollars.
        funded_amnt:  Amount originally requested by the borrower.

    Returns:
        Acceptance probability (0.0–1.0) as a formatted string.
    """
    try:
        model = _get_model()
        prob = model.predict_proba(
            grade=grade,
            offered_rate=offered_rate,
            loan_amnt=loan_amnt,
            annual_inc=annual_inc,
            funded_amnt=funded_amnt,
        )
        return (
            f"P(accept | grade={grade}, rate={offered_rate:.1%}, "
            f"amount=${loan_amnt:,.0f}, income=${annual_inc:,.0f}) = {prob:.3f}"
        )
    except Exception as e:
        return f"ERROR: {e}"
