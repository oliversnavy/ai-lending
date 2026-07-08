
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel

# Load val data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Risk model
model = pickle.load(open('risk_model.pkl', 'rb'))

feature_cols = [
    'grade', 'sub_grade', 'fico_range_low', 'fico_range_high', 'dti',
    'annual_inc', 'loan_amnt', 'funded_amnt', 'term',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util',
    'open_acc', 'emp_length', 'home_ownership', 'verification_status',
    'purpose', 'acc_now_delinq', 'total_acc', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'collections_12_mths_ex_med', 'pub_rec_bankruptcies', 'tax_liens',
    'chargeoff_within_12_mths', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m'
]

for col in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']:
    val[col + '_code'] = pd.Categorical(val[col]).codes

cat_cols = [c + '_code' for c in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']]
num_cols = [c for c in feature_cols if c not in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']]
all_cols = num_cols + cat_cols

X_val = val[all_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
risk_scores = model.predict_proba(X_val)[:, 1]
val['risk_score'] = risk_scores

# Pricing
grade_base_rate = {'C': 0.21, 'D': 0.24, 'E': 0.295, 'F': 0.34}

def get_offer_rate(row):
    grade = row['grade']
    risk = row['risk_score']
    if grade not in grade_base_rate:
        return None
    base = grade_base_rate[grade]
    risk_adj = (risk - 0.5) * 0.04
    burden = row['loan_amnt'] / max(row['annual_inc'], 1.0)
    burden_adj = (burden - 0.3) * 0.05
    term_adj = 0.01 if '60' in str(row['term']) else 0.0
    return np.clip(base + risk_adj + burden_adj + term_adj, 0.21, 0.36)

val['offered_rate'] = val.apply(get_offer_rate, axis=1)

# Acceptance
sens_model = SensitivityModel()
batch = val[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
val['p_accept'] = sens_model.predict_proba_batch(batch)

# Use risk_score as expected default probability
val['expected_default_prob'] = val['risk_score']

# P&L with risk model predicted defaults
val['time_fraction'] = val['observed_time'] / 12.0
val['expected_principal'] = val['p_accept'] * val['loan_amnt']
val['expected_interest'] = val['p_accept'] * val['loan_amnt'] * val['offered_rate'] * val['time_fraction']
val['expected_loss'] = val['p_accept'] * val['loan_amnt'] * val['expected_default_prob']
val['expected_pnl'] = val['expected_interest'] - val['expected_loss']

# Filter valid
valid = val[val['p_accept'].notna() & (val['p_accept'] > 0)].copy()
valid['roi'] = valid['expected_pnl'] / valid['expected_principal'].clip(lower=1)

# Greedy selection
cap = 15_000_000
valid_sorted = valid.sort_values('roi', ascending=False)

selected_indices = []
cumulative = 0.0
for i in range(len(valid_sorted)):
    row = valid_sorted.iloc[i]
    if cumulative + row['expected_principal'] <= cap:
        selected_indices.append(i)
        cumulative += row['expected_principal']
    else:
        break

selected = valid_sorted.iloc[selected_indices]

# Metrics
total_principal = selected['expected_principal'].sum()
total_pnl = selected['expected_pnl'].sum()
loans_funded = selected['p_accept'].sum()
acceptance_rate = selected['p_accept'].mean()

print(f"=== RESULTS (using risk model predicted defaults) ===")
print(f"Total principal: ${total_principal:,.0f}")
print(f"Total P&L: ${total_pnl:,.0f}")
print(f"Loans funded: {loans_funded:.0f}")
print(f"Acceptance rate: {acceptance_rate:.4f}")
print(f"Number of loans: {len(selected):,}")
print(f"ROI: {total_pnl / total_principal:.2%}")

# Check stats
print(f"\nSelected loan stats:")
print(f"  Avg observed time: {selected['observed_time'].mean():.1f} months")
print(f"  Avg offered rate: {selected['offered_rate'].mean():.4f}")
print(f"  Avg risk score: {selected['risk_score'].mean():.4f}")
print(f"  Avg loan amount: ${selected['loan_amnt'].mean():.0f}")
print(f"  Grade distribution:")
print(selected['grade'].value_counts())
