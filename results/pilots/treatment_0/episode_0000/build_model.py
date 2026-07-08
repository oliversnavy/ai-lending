import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

print("Training data shape:", train.shape)
print("Validation data shape:", val.shape)

# Feature engineering - prepare for risk model
def engineer_features(df):
    """Create features for default prediction."""
    df = df.copy()
    
    # Encode grade as numeric
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_num'] = df['grade'].map(grade_map)
    
    # Encode sub_grade
    sub_grade_map = {}
    for g in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        for i in range(1, 6):
            sub_grade_map[f"{g}{i}"] = grade_map[g] + (i-1) * 0.1
    
    # FICO midpoint
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    # DTI ratio
    df['dti_ratio'] = df['dti'] / 100
    
    # Loan to income ratio
    df['lti'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    
    # Revolving utilization
    df['revol_util_clean'] = df['revol_util'].fillna(0)
    if df['revol_util_clean'].dtype == 'object':
        df['revol_util_clean'] = df['revol_util_clean'].str.rstrip('%').astype(float)
    
    # Term as numeric
    df['term_months'] = df['term'].str.replace(' months', '').astype(float)
    
    # Create interaction features
    df['fico_x_grade'] = df['fico_mid'] * df['grade_num']
    df['dti_x_grade'] = df['dti'] * df['grade_num']
    df['lti_x_grade'] = df['lti'] * df['grade_num']
    
    # Income buckets
    df['income_log'] = np.log1p(df['annual_inc'])
    
    # Delinquency indicators
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_inq'] = (df['inq_last_6mths'] > 0).astype(int)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    
    # Recent derogatory
    df['recent_derog'] = (df['mths_since_last_major_derog'].fillna(999) < 36).astype(int)
    
    # Chargeoff history
    df['has_chargeoff'] = (df['chargeoff_within_12_mths'] > 0).astype(int)
    df['has_collections'] = (df['collections_12_mths_ex_med'] > 0).astype(int)
    
    # Number of accounts
    df['total_accounts'] = df['total_acc']
    df['open_accounts'] = df['open_acc']
    
    # Selected features for model
    feature_cols = [
        'grade_num', 'fico_mid', 'fico_range_low', 'fico_range_high',
        'loan_amnt', 'funded_amnt', 'term_months', 'int_rate',
        'dti', 'dti_ratio', 'lti', 'annual_inc', 'income_log',
        'revol_bal', 'revol_util_clean', 'open_acc', 'total_acc',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq',
        'collections_12_mths_ex_med', 'mths_since_last_major_derog',
        'has_delinq', 'has_inq', 'has_pub_rec', 'recent_derog',
        'has_chargeoff', 'has_collections',
        'sub_grade', 'home_ownership', 'verification_status', 'purpose',
        'grade_num', 'fico_x_grade', 'dti_x_grade', 'lti_x_grade',
        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'pub_rec_bankruptcies', 'tax_liens',
        'mo_sin_old_il_acct', 'mo_sin_rcnt_tl',
        'mths_since_recent_bc', 'mths_since_recent_inq',
        'num_accts_ever_120_pd', 'num_rev_accts', 'num_bc_tl', 'num_il_tl',
        'pct_tl_nvr_dlq', 'percent_bc_gt_75',
        'inq_last_12m', 'acc_open_past_24mths',
        'total_bal_il', 'il_util', 'open_il_24m', 'open_il_12m',
        'max_bal_bc', 'all_util', 'tot_hi_cred_lim',
        'bc_open_to_buy', 'bc_util',
        'tot_cur_bal', 'avg_cur_bal',
        'num_op_rev_tl', 'num_rev_accts', 'num_tl_op_past_12m',
        'num_rev_tl_bal_gt_0',
    ]
    
    # Make sure all features exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    return df[existing_cols]

# Prepare features
train_features = engineer_features(train)
val_features = engineer_features(val)

print("Feature columns:", len(train_features.columns))
print("Sample features:")
print(train_features.head())