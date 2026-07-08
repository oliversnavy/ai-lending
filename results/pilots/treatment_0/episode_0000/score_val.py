
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from data_pipeline.sensitivity_model import SensitivityModel

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

# Load and score
with open('risk_model.pkl', 'rb') as f:
    gbc = pickle.load(f)

val_probs = gbc.predict_proba(val_enc.values.astype(float))[:, 1]
val['default_prob'] = val_probs

# Check AUC on val set
val_auc = roc_auc_score(val['event'].values, val_probs)
print(f"Val AUC: {val_auc:.4f}")

# Check grade distribution in val
print("\nVal grade distribution:")
print(val['grade'].value_counts())

# Check default rates by grade in val
print("\nVal default rates by grade:")
for g in sorted(val['grade'].unique()):
    subset = val[val['grade'] == g]
    print(f"  {g}: {subset['event'].mean():.4f} (n={len(subset)})")
