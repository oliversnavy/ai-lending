
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Select features - using only columns that exist
feature_cols = [
    'grade', 'sub_grade', 'fico_range_low', 'fico_range_high', 'dti',
    'annual_inc', 'loan_amnt', 'funded_amnt', 'term',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util',
    'open_acc', 'emp_length', 'home_ownership', 'verification_status',
    'purpose', 'acc_now_delinq', 'total_acc', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'collections_12_mths_ex_med', 'pub_rec_bankruptcies', 'tax_liens',
    'chargeoff_within_12_mths', 'num_accts_ever_120_pd', 'num_tl_120dpd_2m'
]

# Encode categoricals
for col in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']:
    train[col + '_code'] = pd.Categorical(train[col]).codes

cat_cols = [c + '_code' for c in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']]
num_cols = [c for c in feature_cols if c not in ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose']]

all_cols = num_cols + cat_cols
print("Features:", len(all_cols))

# Prepare data
X = train[all_cols].fillna(0)

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train logistic regression
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, C=0.1))
])

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, train['event'], cv=5, scoring='roc_auc')
print(f"CV ROC AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X, train['event'])
preds = model.predict_proba(X)[:, 1]
print(f"Train ROC AUC: {roc_auc_score(train['event'], preds):.4f}")

# Save the model
import pickle
pickle.dump(model, open('risk_model.pkl', 'wb'))
print("Model saved.")
