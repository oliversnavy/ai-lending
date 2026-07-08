import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Create derived features on both
for df in [train, val]:
    df['term_num'] = (df['term'] == '60 months').astype(int)
    df['grade_num'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

feature_cols = [
    'loan_amnt', 'funded_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util',
    'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'total_acc', 'num_tl_120dpd_2m', 'num_tl_30dpd',
    'fico_range_low', 'fico_range_high', 'term_num', 'grade_num'
]

# Score validation set
risk_model = pickle.load(open('risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

X_val = val[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_val[col] = X_val[col].fillna(med)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

val['default_prob'] = risk_model.predict_proba(scaler.transform(X_val))[:, 1]

# Load sensitivity model
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Test sensitivity model
print("--- Sensitivity model test ---")
test_cases = [
    ('C', 0.21, 10000, 50000, 10000),
    ('C', 0.25, 10000, 50000, 10000),
    ('C', 0.30, 10000, 50000, 10000),
    ('D', 0.21, 10000, 50000, 10000),
    ('D', 0.30, 10000, 50000, 10000),
    ('D', 0.36, 10000, 50000, 10000),
    ('E', 0.25, 10000, 50000, 10000),
    ('E', 0.36, 10000, 50000, 10000),
    ('F', 0.36, 10000, 50000, 10000),
]

for grade, rate, la, ai, fa in test_cases:
    prob = sens_model.predict_proba([[grade, rate, la, ai, fa]])[0, 1]
    print(f"  {grade} @ {rate:.0%}: p_accept={prob:.4f}")
