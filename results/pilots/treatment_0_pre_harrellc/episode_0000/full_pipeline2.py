
import pandas as pd
import numpy as np
import pickle
from lifelines.utils import concordance_index

# Load model and data
model = pickle.load(open('risk_model.pkl', 'rb'))
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Prepare features
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

c_index = concordance_index(val['observed_time'], -risk_scores, val['event'])
print(f"C-index: {c_index:.4f}")

# Better pricing strategy
# Grade-based base rates: offer at or near market rate
# Then adjust for risk and loan burden

grade_base_rate = {
    'C': 0.21,    # 2pp above market (19%)
    'D': 0.24,    # at market (24%)
    'E': 0.295,   # at market (29.5%)
    'F': 0.34     # at market (34%)
}

def get_offer_rate(row):
    grade = row['grade']
    risk = row['risk_score']
    
    if grade not in grade_base_rate:
        return None
    
    base = grade_base_rate[grade]
    
    # Risk adjustment: higher risk borrowers get higher rates
    # Scale: risk 0.3 -> -0.02, risk 0.5 -> 0, risk 0.7 -> +0.02
    risk_adj = (risk - 0.5) * 0.04
    
    # Loan burden adjustment: high loan/income ratio -> higher rate
    burden = row['loan_amnt'] / max(row['annual_inc'], 1.0)
    burden_adj = (burden - 0.3) * 0.05
    
    # Term adjustment: longer terms slightly higher rate
    term_adj = 0.0 if '36' in str(row['term']) else 0.01
    
    offered_rate = base + risk_adj + burden_adj + term_adj
    
    # Clamp to 0.21-0.36
    offered_rate = np.clip(offered_rate, 0.21, 0.36)
    
    return offered_rate

val['offered_rate'] = val.apply(get_offer_rate, axis=1)

# Check stats
print(f"\nLoans offered: {(val['offered_rate'].notna()).sum():,}")
print(f"Offered rate by grade:")
for g in ['C', 'D', 'E', 'F']:
    subset = val[val['grade'] == g]
    offered = subset['offered_rate'].dropna()
    if len(offered) > 0:
        print(f"  {g}: min={offered.min():.4f}, max={offered.max():.4f}, mean={offered.mean():.4f}, median={offered.median():.4f}")

# Save for next step
val.to_pickle('val_scored.pkl')
print("\nSaved scored validation data.")
