
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Encode grade as numeric
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
train['grade_num'] = train['grade'].map(grade_map)
train['sub_grade_num'] = train['sub_grade'].str[-1].astype(float)
train['term_num'] = train['term'].str.extract(r'(\d+)')[0].astype(float)

feature_cols = [
    'grade_num', 'sub_grade_num', 'fico_range_low', 'fico_range_high',
    'term_num', 'loan_amnt', 'funded_amnt', 'annual_inc', 'dti',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util',
    'open_acc', 'total_acc', 'acc_now_delinq', 'collections_12_mths_ex_med',
    'chargeoff_within_12_mths', 'tax_liens', 'pub_rec_bankruptcies',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'pct_tl_nvr_dlq', 'total_rev_hi_lim', 'mths_since_recent_bc'
]

train_enc = train[feature_cols].copy()
for col in train_enc.columns:
    train_enc[col] = pd.to_numeric(train_enc[col], errors='coerce')
train_enc = train_enc.fillna(0)

y = train['event'].values
X = train_enc.values.astype(float)

gbc = GradientBoostingClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    min_samples_leaf=200, subsample=0.8, random_state=42
)
gbc.fit(X, y)

# Save model
with open('risk_model.pkl', 'wb') as f:
    pickle.dump(gbc, f)

print("Model saved. Train AUC:", roc_auc_score(y, gbc.predict_proba(X)[:, 1]))
