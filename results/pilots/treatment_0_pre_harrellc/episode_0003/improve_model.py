import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import pickle
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
    'acc_now_delinq', 'total_acc', 'fico_range_low', 'fico_range_high',
    'term_num', 'grade_num',
    'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_recent_bc',
    'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
    'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
    'num_sats', 'num_tl_op_past_12m',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens',
    'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit',
    'revol_bal_joint', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
    'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
    'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
    'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'tot_coll_amt', 'tot_cur_bal',
]

feature_cols = list(dict.fromkeys(feature_cols))
feature_cols = [c for c in feature_cols if c in train.columns]

X_train = train[feature_cols].copy()
y_train = train['event'].copy()
X_val = val[feature_cols].copy()
y_val = val['event'].copy()

for col in feature_cols:
    med = X_train[col].median()
    X_train[col] = X_train[col].fillna(med)
    X_val[col] = X_val[col].fillna(med)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("=== Model Comparison ===")

lr = LogisticRegression(max_iter=1000, C=0.1)
lr.fit(X_train_scaled, y_train)
lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:, 1])
print(f"Logistic Regression: AUC={lr_auc:.4f}")

gb = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
gb_auc = roc_auc_score(y_val, gb.predict_proba(X_val)[:, 1])
print(f"Gradient Boosting: AUC={gb_auc:.4f}")

if gb_auc > lr_auc:
    print("\nUsing Gradient Boosting")
    best_model = gb
    scaler_used = None
else:
    print("\nUsing Logistic Regression")
    best_model = lr
    scaler_used = scaler

y_pred = best_model.predict_proba(X_val)[:, 1] if scaler_used is None else best_model.predict_proba(X_val_scaled)[:, 1]
ci = concordance_index(val['observed_time'], -y_pred, val['event'])
print(f"Concordance Index: {ci:.4f}")

if scaler_used is not None:
    pickle.dump(best_model, open('risk_model.pkl', 'wb'))
    pickle.dump(scaler_used, open('scaler.pkl', 'wb'))
else:
    pickle.dump(best_model, open('risk_model.pkl', 'wb'))
    import os
    if os.path.exists('scaler.pkl'):
        os.remove('scaler.pkl')

print(f"\nModel saved. AUC={gb_auc if gb_auc > lr_auc else lr_auc:.4f}, CI={ci:.4f}")
