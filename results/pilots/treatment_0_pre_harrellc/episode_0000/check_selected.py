
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel

# Load val data fresh
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

# P&L
val['time_fraction'] = val['observed_time'] / 12.0
val['expected_principal'] = val['p_accept'] * val['loan_amnt']
val['expected_interest'] = val['p_accept'] * val['loan_amnt'] * val['offered_rate'] * val['time_fraction']
val['expected_loss'] = val['p_accept'] * val['loan_amnt'] * val['event']
val['expected_pnl'] = val['expected_interest'] - val['expected_loss']

# Filter valid
valid = val[val['p_accept'].notna() & (val['p_accept'] > 0)].copy()

# Check a sample of selected loans
valid['roi'] = valid['expected_pnl'] / valid['expected_principal'].clip(lower=1)
valid_sorted = valid.sort_values('roi', ascending=False)

# Select top 20
top20 = valid_sorted.head(20)
print("Top 20 loans by ROI:")
for _, row in top20.iterrows():
    time_frac = row['observed_time'] / 12
    interest = row['p_accept'] * row['loan_amnt'] * row['offered_rate'] * time_frac
    loss = row['p_accept'] * row['loan_amnt'] * row['event']
    pnl = interest - loss
    print(f"  Grade={row['grade']}, Rate={row['offered_rate']:.2%}, Amt={row['loan_amnt']:.0f}, "
          f"Time={row['observed_time']:.1f}mo, Accept={row['p_accept']:.3f}, "
          f"Interest={interest:.0f}, Loss={loss:.0f}, P&L={pnl:.0f}, "
          f"ROI={pnl/(row['p_accept']*row['loan_amnt']):.2%}")

# Check stats of selected
selected = valid_sorted.head(1362)
print(f"\n=== Selected 1362 loans ===")
print(f"Avg observed time: {selected['observed_time'].mean():.1f} months")
print(f"Avg offered rate: {selected['offered_rate'].mean():.4f}")
print(f"Avg p_accept: {selected['p_accept'].mean():.4f}")
print(f"Avg loan amount: ${selected['loan_amnt'].mean():.0f}")
print(f"Avg default rate: {selected['event'].mean():.4f}")

# Manual P&L check
avg_time = selected['observed_time'].mean() / 12
avg_rate = selected['offered_rate'].mean()
avg_default = selected['event'].mean()
avg_loan = selected['loan_amnt'].mean()
avg_accept = selected['p_accept'].mean()

expected_interest = avg_accept * avg_loan * avg_rate * avg_time
expected_loss = avg_accept * avg_loan * avg_default
pnl_per_loan = expected_interest - expected_loss
total_principal = avg_accept * avg_loan * len(selected)
total_pnl = pnl_per_loan * len(selected)

print(f"\nManual calculation:")
print(f"  Avg time: {avg_time:.2f} years")
print(f"  Expected interest per loan: ${expected_interest:.0f}")
print(f"  Expected loss per loan: ${expected_loss:.0f}")
print(f"  P&L per loan: ${pnl_per_loan:.0f}")
print(f"  Expected principal per loan: ${avg_accept * avg_loan:.0f}")
print(f"  Total principal: ${total_principal:,.0f}")
print(f"  Total P&L: ${total_pnl:,.0f}")
