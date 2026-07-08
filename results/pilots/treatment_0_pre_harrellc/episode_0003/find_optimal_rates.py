import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

for df in [train, val]:
    df['term_num'] = (df['term'] == '60 months').astype(int)
    df['grade_num'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

feature_cols = [
    'loan_amnt', 'funded_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util',
    'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'total_acc', 'num_tl_120dpd_2m', 'num_tl_30dpd',
    'fico_range_low', 'fico_range_high', 'term_num', 'grade_num'
]

risk_model = pickle.load(open('risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Score val
X_val = val[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_val[col] = X_val[col].fillna(med)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
val['default_prob'] = risk_model.predict_proba(scaler.transform(X_val))[:, 1]

# Filter to viable segment
val_viable = val[val['grade'].isin(['C', 'D', 'E', 'F', 'G'])].copy()
print(f"Viable applicants: {len(val_viable)}")

# --- Optimized pricing function ---
# Strategy: For each applicant, compute the rate that maximizes expected P&L
# E[P&L] = p_accept(rate) * [loan_amnt * rate * (obs_time/12) - loan_amnt * default_prob]
# 
# We need to find the rate that maximizes this. Since p_accept decreases with rate,
# there's a tradeoff. We use a parametric approach:
# rate = base_rate + risk_adjustment * default_prob + loan_size_adjustment

# First, let's do a coarse grid search on a sample to find optimal rates
sample = val_viable.sample(50000, random_state=42)

# Compute default probs for sample
X_sample = sample[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_sample[col] = X_sample[col].fillna(med)
X_sample = X_sample.replace([np.inf, -np.inf], np.nan).fillna(0)
sample['default_prob'] = risk_model.predict_proba(scaler.transform(X_sample))[:, 1]

# For each grade, find the rate that maximizes expected P&L
results = []
for grade in ['C', 'D', 'E', 'F', 'G']:
    grade_data = sample[sample['grade'] == grade]
    if len(grade_data) < 100:
        continue
    
    best_pnl = -np.inf
    best_rate = None
    
    for rate in np.arange(0.21, 0.37, 0.01):
        grade_data['offered_rate'] = rate
        offer_df = grade_data[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
        p_accept = sens_model.predict_proba_batch(offer_df)
        
        # Expected P&L per applicant
        expected_pnl = p_accept * (grade_data['loan_amnt'] * rate * (grade_data['observed_time'] / 12.0) - grade_data['loan_amnt'] * grade_data['default_prob'])
        mean_pnl = expected_pnl.mean()
        
        if mean_pnl > best_pnl:
            best_pnl = mean_pnl
            best_rate = rate
    
    results.append({'grade': grade, 'best_rate': best_rate, 'best_pnl': best_pnl})
    print(f"Grade {grade}: best_rate={best_rate:.2f}, best_pnl_per_loan=${best_pnl:.2f}")

print("\n=== Grid Search Results ===")
for r in results:
    print(f"  {r['grade']}: rate={r['best_rate']:.2f}")
