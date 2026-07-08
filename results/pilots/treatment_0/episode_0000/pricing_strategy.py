
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

# Build pricing function
model = SensitivityModel()

def get_offered_rate(grade, default_prob):
    base_rates = {
        'A': 0.21, 'B': 0.21,
        'C': 0.21,
        'D': 0.25,
        'E': 0.21,
        'F': 0.35,
        'G': 0.30,
    }
    base = base_rates.get(grade, 0.21)
    
    if default_prob > 0.35:
        base += 0.05
    elif default_prob > 0.25:
        base += 0.03
    
    return min(base, 0.50)

# Create full test dataframe with required columns
test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob']].copy()
test_df['offered_rate'] = test_df.apply(
    lambda row: get_offered_rate(row['grade'], row['default_prob']), axis=1
)

# Calculate acceptance probabilities
acceptance_probs = model.predict_proba_batch(test_df)
test_df['acceptance_prob'] = acceptance_probs

# Check acceptance by grade
print("Acceptance rates by grade:")
for grade in ['C', 'D', 'E', 'F', 'G']:
    subset = test_df[test_df['grade'] == grade]
    print(f"  {grade}: {subset['acceptance_prob'].mean():.1%} (n={len(subset)}, rate={subset['offered_rate'].mean():.0%})")

# Filter to viable segment (acceptance > 0)
viable = test_df[test_df['acceptance_prob'] > 0.01]
print(f"\nTotal viable applicants: {len(viable)}")
print(f"Viable grades: {viable['grade'].value_counts().to_dict()}")
