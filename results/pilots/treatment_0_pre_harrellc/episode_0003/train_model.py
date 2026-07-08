import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Feature engineering
feature_cols = [
    'loan_amnt', 'funded_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util',
    'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'total_acc', 'num_tl_120dpd_2m', 'num_tl_30dpd',
    'fico_range_low', 'fico_range_high',
]

# Encode term as numeric
train['term_num'] = (train['term'] == '60 months').astype(int)
val['term_num'] = (val['term'] == '60 months').astype(int)

feature_cols.append('term_num')

# Encode grade as numeric
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

target = 'event'

train['grade_num'] = train['grade'].map(grade_map)
val['grade_num'] = val['grade'].map(grade_map)

feature_cols.append('grade_num')

X_train = train[feature_cols].copy()
y_train = train[target].copy()

X_val = val[feature_cols].copy()
y_val = val[target].copy()

# Fill missing values with median per column
for col in feature_cols:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

# Replace infinities
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train logistic regression - multiple C values
for C_val in [0.01, 0.1, 1.0, 10.0]:
    model = LogisticRegression(max_iter=1000, C=C_val)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"C={C_val}: AUC={auc:.4f}")

# Use best C
best_C = 0.1
model = LogisticRegression(max_iter=1000, C=best_C)
model.fit(X_train_scaled, y_train)
y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)
print(f"\nFinal model AUC: {auc:.4f}")

# Save model
import pickle
pickle.dump(model, open('risk_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("\nFeature importances:")
for feat, imp in sorted(zip(feature_cols, model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {feat}: {imp:.4f}")
