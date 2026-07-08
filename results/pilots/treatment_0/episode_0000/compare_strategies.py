
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

# Test different pricing strategies
strategies = {}

# Strategy 1: Fixed rate per grade (current)
def rate_1(grade, dp):
    base = {'C': 0.21, 'D': 0.25, 'E': 0.21, 'F': 0.35, 'G': 0.30}.get(grade, 0.21)
    if dp > 0.35: base += 0.05
    elif dp > 0.25: base += 0.03
    return min(base, 0.50)

# Strategy 2: All at 21% (minimum viable)
def rate_2(grade, dp):
    return 0.21

# Strategy 3: Offer at market rate for each grade
def rate_3(grade, dp):
    market = {'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}.get(grade, 0.21)
    return max(market, 0.21)

# Strategy 4: Higher rates for higher risk
def rate_4(grade, dp):
    base = {'C': 0.25, 'D': 0.30, 'E': 0.30, 'F': 0.35, 'G': 0.35}.get(grade, 0.21)
    base += dp * 0.10  # risk premium
    return min(base, 0.50)

# Strategy 5: Conservative - only offer to low-risk borrowers
def rate_5(grade, dp):
    base = {'C': 0.21, 'D': 0.25, 'E': 0.21, 'F': 0.35, 'G': 0.30}.get(grade, 0.21)
    if dp > 0.20: return None  # Don't offer to high-risk
    return base

for name, rate_fn in [('1_graded', rate_1), ('2_flat21', rate_2), ('3_market', rate_3), 
                       ('4_risk_premium', rate_4), ('5_conservative', rate_5)]:
    test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob', 
                   'event', 'observed_time']].copy()
    test_df['offered_rate'] = test_df.apply(
        lambda row: rate_fn(row['grade'], row['default_prob']) if rate_fn(row['grade'], row['default_prob']) else 0.21, 
        axis=1
    )
    
    # Filter out None rates
    test_df = test_df[test_df['offered_rate'] > 0]
    
    acceptance_probs = model.predict_proba_batch(test_df)
    test_df['acceptance_prob'] = acceptance_probs
    
    # Calculate P&L
    test_df['expected_interest'] = test_df['loan_amnt'] * test_df['offered_rate'] * test_df['observed_time'] / 12
    test_df['expected_loss'] = test_df['loan_amnt'] * test_df['default_prob']
    test_df['expected_pnl'] = test_df['acceptance_prob'] * (test_df['expected_interest'] - test_df['expected_loss'])
    test_df['expected_principal'] = test_df['loan_amnt'] * test_df['acceptance_prob']
    
    # Sort and cap
    sorted_df = test_df.sort_values('expected_pnl', ascending=False)
    cumulative = sorted_df['expected_principal'].cumsum()
    selected = sorted_df[cumulative <= 15000000]
    
    expected_loans = selected['acceptance_prob'].sum()
    pnl = selected['expected_pnl'].sum()
    principal = selected['expected_principal'].sum()
    
    strategies[name] = {
        'pnl': pnl,
        'loans': expected_loans,
        'principal': principal,
        'acceptance_rate': selected['acceptance_prob'].mean(),
        'n_selected': len(selected)
    }
    
    print(f"{name}: P&L=${pnl:,.0f}, Loans={expected_loans:.0f}, Principal=${principal:,.0f}, "
          f"Acceptance={selected['acceptance_prob'].mean():.1%}, N={len(selected)}")
