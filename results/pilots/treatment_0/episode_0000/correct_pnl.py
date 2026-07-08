
import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Encode for model
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

for df in [train, val]:
    df['grade_num'] = df['grade'].map(grade_map)
    df['sub_grade_num'] = df['sub_grade'].str[-1].astype(float)
    df['term_num'] = df['term'].str.extract(r'(\d+)')[0].astype(float)

feature_cols = [
    'grade_num', 'sub_grade_num', 'fico_range_low', 'fico_range_high',
    'term_num', 'loan_amnt', 'funded_amnt', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util',
    'open_acc', 'total_acc', 'acc_now_delinq', 'collections_12_mths_ex_med',
    'chargeoff_within_12_mths', 'tax_liens', 'pub_rec_bankruptcies',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'pct_tl_nvr_dlq', 'total_rev_hi_lim', 'mths_since_recent_bc'
]

# Score validation data
val_enc = val[feature_cols].copy()
for col in val_enc.columns:
    val_enc[col] = pd.to_numeric(val_enc[col], errors='coerce')
val_enc = val_enc.fillna(0)

with open('risk_model.pkl', 'rb') as f:
    gbc = pickle.load(f)

val_probs = gbc.predict_proba(val_enc.values.astype(float))[:, 1]
val['default_prob'] = val_probs

model = SensitivityModel()

# Correct P&L calculation using actual outcomes
def evaluate_strategy(name, rate_fn, grade_filter=None):
    test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob', 
                   'event', 'observed_time']].copy()
    
    if grade_filter:
        test_df = test_df[test_df['grade'].isin(grade_filter)]
    
    test_df['offered_rate'] = test_df.apply(
        lambda row: rate_fn(row['grade'], row['default_prob']), axis=1
    )
    test_df = test_df[test_df['offered_rate'] > 0]
    
    acceptance_probs = model.predict_proba_batch(test_df)
    test_df['acceptance_prob'] = acceptance_probs
    
    # Correct P&L: use actual event (not default_prob) for principal_lost
    test_df['interest_collected'] = test_df['loan_amnt'] * test_df['offered_rate'] * test_df['observed_time'] / 12
    test_df['principal_lost'] = test_df['loan_amnt'] * test_df['event']
    test_df['actual_pnl'] = test_df['interest_collected'] - test_df['principal_lost']
    test_df['expected_pnl'] = test_df['acceptance_prob'] * test_df['actual_pnl']
    test_df['expected_principal'] = test_df['loan_amnt'] * test_df['acceptance_prob']
    
    sorted_df = test_df.sort_values('expected_pnl', ascending=False)
    cumulative = sorted_df['expected_principal'].cumsum()
    selected = sorted_df[cumulative <= 15000000]
    
    expected_loans = selected['acceptance_prob'].sum()
    pnl = selected['expected_pnl'].sum()
    principal = selected['expected_principal'].sum()
    
    return {
        'name': name,
        'pnl': pnl,
        'loans': expected_loans,
        'principal': principal,
        'acceptance_rate': selected['acceptance_prob'].mean(),
        'n_selected': len(selected)
    }

# Strategy B: Very high rates
def rate_B(grade, dp):
    base = {'C': 0.30, 'D': 0.35, 'E': 0.35, 'F': 0.40, 'G': 0.40}.get(grade, 0.21)
    base += dp * 0.05
    return min(max(base, 0.21), 0.50)

# Strategy A: Moderate rates
def rate_A(grade, dp):
    base = {'C': 0.25, 'D': 0.30, 'E': 0.30, 'F': 0.35, 'G': 0.35}.get(grade, 0.21)
    base += dp * 0.15
    return min(max(base, 0.21), 0.50)

# Strategy C: Market rate
def rate_C(grade, dp):
    market = {'C': 0.21, 'D': 0.25, 'E': 0.30, 'F': 0.35, 'G': 0.37}.get(grade, 0.21)
    base = max(market, 0.21)
    base += dp * 0.08
    return min(base, 0.50)

# Strategy D: Flat 21%
def rate_D(grade, dp):
    return 0.21

# Strategy E: Only C and D grades
def rate_E(grade, dp):
    if grade not in ['C', 'D']:
        return 0
    base = {'C': 0.21, 'D': 0.25}.get(grade, 0.21)
    base += dp * 0.05
    return min(base, 0.50)

strategies = [
    ('B_high', rate_B),
    ('A_moderate', rate_A),
    ('C_market', rate_C),
    ('D_flat', rate_D),
    ('E_CD_only', rate_E),
]

for name, fn in strategies:
    result = evaluate_strategy(name, fn)
    print(f"{name}: P&L=${result['pnl']:,.0f}, Loans={result['loans']:.0f}, "
          f"Principal=${result['principal']:,.0f}, Acceptance={result['acceptance_rate']:.1%}")
