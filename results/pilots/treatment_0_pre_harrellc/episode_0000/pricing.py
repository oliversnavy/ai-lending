
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

# Load model and data
model = pickle.load(open('risk_model.pkl', 'rb'))
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Prepare features for scoring
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

# Concordance index using lifelines or sklearn
# Let's use a simple approach - compute C-index manually
from lifelines.utils import concordance_index

# Create survival data
c_index = concordance_index(
    val['observed_time'], 
    -risk_scores,  # negative because higher risk = shorter survival
    val['event']
)
print(f"C-index: {c_index:.4f}")

# Now design pricing function
# Strategy: grade-based pricing with risk adjustment
# C: 21-24%, D: 24-28%, E: 28-32%, F: 32-36%
# Plus risk adjustment within grade

val['term_months'] = val['term'].apply(lambda x: 36 if '36' in str(x) else 60)
val['time_fraction'] = val['observed_time'] / 12.0

# Base rates by grade
grade_base_rate = {
    'C': 0.21,
    'D': 0.25,
    'E': 0.30,
    'F': 0.35
}

# For each applicant, set offered rate
def get_offer_rate(row):
    grade = row['grade']
    risk = row['risk_score']
    
    if grade not in grade_base_rate:
        return None  # Don't offer to A, B, G
    
    base = grade_base_rate[grade]
    
    # Risk adjustment: higher risk = higher rate
    # Scale risk from 0-1 to +/- 0.05
    adjustment = (risk - 0.5) * 0.10
    
    offered_rate = base + adjustment
    
    # Clamp to 0.21-0.36
    offered_rate = np.clip(offered_rate, 0.21, 0.36)
    
    return offered_rate

val['offered_rate'] = val.apply(get_offer_rate, axis=1)

# Check how many loans we'd offer
print(f"\nLoans offered (rate between 21-36%): {(val['offered_rate'].notna()).sum()}")
print(f"Median offered rate: {val['offered_rate'].median():.4f}")
print(f"Offered rate distribution:")
print(val['offered_rate'].describe())
