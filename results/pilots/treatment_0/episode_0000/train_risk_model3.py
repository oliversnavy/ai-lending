import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Feature engineering - simpler version
def engineer_features(df):
    df = df.copy()
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_num'] = df['grade'].map(grade_map)
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['dti_ratio'] = df['dti'] / 100
    df['lti'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['term_months'] = df['term'].str.replace(' months', '').astype(float)
    df['income_log'] = np.log1p(df['annual_inc'])
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_inq'] = (df['inq_last_6mths'] > 0).astype(int)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    df['has_chargeoff'] = (df['chargeoff_within_12_mths'] > 0).astype(int)
    df['has_collections'] = (df['collections_12_mths_ex_med'] > 0).astype(int)
    
    # Encode categorical features
    df['sub_grade_num'] = df['sub_grade'].str[0].map(grade_map) + df['sub_grade'].str[1:].astype(float) / 10
    df['home_own_num'] = df['home_ownership'].map({'OTHER': 0, 'NONE': 1, 'MORTGAGE': 2, 'RENT': 3, 'OWN': 4})
    df['verify_num'] = df['verification_status'].map({'Not Verified': 0, 'Source Verified': 1, 'Verified': 2})
    df['purpose_num'] = df['purpose'].map({
        'car': 1, 'credit_card': 2, 'education': 3, 'home_improvement': 4,
        'major_purchase': 5, 'medical': 6, 'moving': 7, 'other': 8,
        'debt_consolidation': 9, 'renewable_credit': 10, 'wedding': 11
    })
    
    feature_cols = [
        'grade_num', 'sub_grade_num', 'fico_mid', 'fico_range_low', 'fico_range_high',
        'loan_amnt', 'funded_amnt', 'term_months', 'int_rate',
        'dti', 'dti_ratio', 'lti', 'annual_inc', 'income_log',
        'revol_bal', 'revol_util_clean' if 'revol_util_clean' in df.columns else 'revol_util',
        'open_acc', 'total_acc',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq',
        'collections_12_mths_ex_med', 'mths_since_last_major_derog',
        'has_delinq', 'has_inq', 'has_pub_rec',
        'has_chargeoff', 'has_collections',
        'home_own_num', 'verify_num', 'purpose_num',
        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'pub_rec_bankruptcies', 'tax_liens',
        'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_inq',
        'num_accts_ever_120_pd', 'num_rev_accts', 'num_bc_tl', 'num_il_tl',
        'pct_tl_nvr_dlq', 'percent_bc_gt_75',
        'inq_last_12m', 'acc_open_past_24mths',
        'total_bal_il', 'il_util', 'open_il_24m', 'open_il_12m',
        'max_bal_bc', 'all_util', 'tot_hi_cred_lim',
        'bc_open_to_buy', 'bc_util',
        'tot_cur_bal', 'avg_cur_bal',
        'num_op_rev_tl', 'num_tl_op_past_12m',
        'num_rev_tl_bal_gt_0',
    ]
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    return df[existing_cols]

train_features = engineer_features(train)
val_features = engineer_features(val)

# Fill NaN with 0 and convert
X_train = train_features.fillna(0).values.astype(float)
X_val = val_features.fillna(0).values.astype(float)

y_train = train['event'].values
y_val = val['event'].values

print("Features:", X_train.shape[1])

# Train a simpler model with fewer estimators
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f"\nValidation AUC: {auc:.4f}")

# Check performance by grade
val_features['pred_prob'] = y_pred
for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    mask = val['grade'] == grade
    grade_auc = roc_auc_score(val.loc[mask, 'event'], val.loc[mask, 'pred_prob'])
    print(f"  {grade}: AUC={grade_auc:.4f}, n={mask.sum()}, default_rate={val.loc[mask, 'event'].mean():.3f}")

import pickle
with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved")