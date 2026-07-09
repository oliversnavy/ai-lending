from __future__ import annotations
import pathlib
import pickle

from langchain_core.tools import tool

_MODEL_BAD  = None
_MODEL_GOOD = None
_PATH_BAD   = pathlib.Path("data/processed/sensitivity_model_defaulter.pkl")
_PATH_GOOD  = pathlib.Path("data/processed/sensitivity_model_nondefaulter.pkl")


def _get_models():
    global _MODEL_BAD, _MODEL_GOOD
    if _MODEL_BAD is None:
        with open(_PATH_BAD, "rb") as f:
            _MODEL_BAD = pickle.load(f)
    if _MODEL_GOOD is None:
        with open(_PATH_GOOD, "rb") as f:
            _MODEL_GOOD = pickle.load(f)
    return _MODEL_BAD, _MODEL_GOOD


@tool
def sensitivity_model_query(
    grade: str,
    offered_rate: float,
    loan_amnt: float,
    annual_inc: float,
    funded_amnt: float,
) -> str:
    """
    Spot-check acceptance probabilities for a single borrower + offer.
    Returns P(accept) for both the defaulter and non-defaulter sensitivity models.

    For bulk evaluation across many applicants use code_executor instead:
        import pickle
        model_bad  = pickle.load(open('data/processed/sensitivity_model_defaulter.pkl',  'rb'))
        model_good = pickle.load(open('data/processed/sensitivity_model_nondefaulter.pkl','rb'))
        p_accept_bad  = model_bad.predict_proba_batch(df)
        p_accept_good = model_good.predict_proba_batch(df)
        # df needs columns: grade, offered_rate, loan_amnt, annual_inc, funded_amnt

    Args:
        grade:        LendingClub grade letter (A–G).
        offered_rate: Annual interest rate offered, as a decimal (e.g. 0.24 for 24%).
        loan_amnt:    Dollar amount offered.
        annual_inc:   Borrower annual income in dollars.
        funded_amnt:  Amount originally requested by the borrower.

    Returns:
        Both acceptance probabilities as a formatted string.
    """
    try:
        model_bad, model_good = _get_models()
        kwargs = dict(
            grade=grade,
            offered_rate=offered_rate,
            loan_amnt=loan_amnt,
            annual_inc=annual_inc,
            funded_amnt=funded_amnt,
        )
        p_bad  = model_bad.predict_proba(**kwargs)
        p_good = model_good.predict_proba(**kwargs)
        return (
            f"grade={grade}, rate={offered_rate:.1%}, amount=${loan_amnt:,.0f}, "
            f"income=${annual_inc:,.0f}\n"
            f"  P(accept | defaulter)     = {p_bad:.3f}  [credit-constrained, rate-inelastic]\n"
            f"  P(accept | non-defaulter) = {p_good:.3f}  [has outside options, rate-sensitive]\n"
            f"  Note: blend with your p_default_hat estimate for effective P(accept)"
        )
    except Exception as e:
        return f"ERROR: {e}"
