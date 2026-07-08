
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
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

# C-index
c_index = concordance_index(val['observed_time'], -risk_scores, val['event'])
print(f"C-index: {c_index:.4f}")

# Now design pricing function with understanding of sensitivity model
# Market rates: A=9.5%, B=14%, C=19%, D=24%, E=29.5%, F=34%, G=37%
# Beta_spread = 12.0, so each 1pp above market reduces log-odds by 12
# At 5pp above market: sigmoid(0.20 - 12*0.05) = sigmoid(-0.40) ≈ 40%
# At 10pp above market: sigmoid(0.20 - 12*0.10) = sigmoid(-1.20) ≈ 23%

# Strategy: offer at market rate or slightly above for each grade
grade_market_rate = {
    'C': 0.19,
    'D': 0.24,
    'E': 0.295,
    'F': 0.34
}

def get_offer_rate(row):
    grade = row['grade']
    risk = row['risk_score']
    
    if grade not in grade_market_rate:
        return None
    
    market = grade_market_rate[grade]
    
    # Risk adjustment: higher risk = higher rate
    # Scale risk from 0-1 to +/- 0.03
    adjustment = (risk - 0.5) * 0.06
    
    # Also consider loan/income burden
    burden = row['loan_amnt'] / max(row['annual_inc'], 1.0)
    burden_adj = (burden - 0.3) * 0.10  # slight adjustment for burden
    
    offered_rate = market + adjustment + burden_adj
    
    # Clamp to 0.21-0.36
    offered_rate = np.clip(offered_rate, 0.21, 0.36)
    
    return offered_rate

val['offered_rate'] = val.apply(get_offer_rate, axis=1)

# Check stats
print(f"\nLoans offered (rate between 21-36%): {(val['offered_rate'].notna()).sum():,}")
print(f"Median offered rate: {val['offered_rate'].median():.4f}")
print(f"Offered rate by grade:")
for g in ['C', 'D', 'E', 'F']:
    subset = val[val['grade'] == g]
    offered = subset['offered_rate'].dropna()
    if len(offered) > 0:
        print(f"  {g}: min={offered.min():.4f}, max={offered.max():.4f}, mean={offered.mean():.4f}")
