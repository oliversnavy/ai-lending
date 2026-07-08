
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel

# Quick re-run to check observed_time
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check observed_time stats
print("Observed time distribution:")
print(f"  Min: {val['observed_time'].min():.1f}")
print(f"  25th: {val['observed_time'].quantile(0.25):.1f}")
print(f"  Median: {val['observed_time'].quantile(0.5):.1f}")
print(f"  75th: {val['observed_time'].quantile(0.75):.1f}")
print(f"  Max: {val['observed_time'].max():.1f}")

# Check term distribution
print(f"\nTerm distribution:")
print(val['term'].value_counts())

# For 36-month loans, check observed_time
term36 = val[val['term'].str.contains('36')]
term60 = val[val['term'].str.contains('60')]
print(f"\n36-month loans - observed time: median={term36['observed_time'].median():.1f}, mean={term36['observed_time'].mean():.1f}")
print(f"60-month loans - observed time: median={term60['observed_time'].median():.1f}, mean={term60['observed_time'].mean():.1f}")

# Check the selected loans
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
val['risk_score'] = risk_scores

sens_model = SensitivityModel()
batch = val[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
val['p_accept'] = sens_model.predict_proba_batch(batch)

valid = val[val['p_accept'].notna() & (val['p_accept'] > 0)].copy()
valid['roi'] = valid['expected_pnl'] / valid['expected_principal'].clip(lower=1) if 'expected_pnl' in valid.columns else valid['expected_pnl'] / valid['loan_amnt'].clip(lower=1)

# Check a few selected loans
valid_sorted = valid.sort_values('roi', ascending=False)
top10 = valid_sorted.head(10)
print(f"\nTop 10 selected loans:")
for _, row in top10.iterrows():
    time_frac = row['observed_time'] / 12
    interest = row['p_accept'] * row['loan_amnt'] * row['offered_rate'] * time_frac
    loss = row['p_accept'] * row['loan_amnt'] * row['event']
    pnl = interest - loss
    print(f"  Grade={row['grade']}, Rate={row['offered_rate']:.2%}, Amt={row['loan_amnt']:.0f}, "
          f"Time={row['observed_time']:.1f}mo, Accept={row['p_accept']:.3f}, "
          f"Interest={interest:.0f}, Loss={loss:.0f}, P&L={pnl:.0f}")
