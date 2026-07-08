import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Check which of my feature columns have NaN
feature_cols = [
    'loan_amnt', 'funded_amnt', 'int_rate', 'grade', 'sub_grade',
    'fico_range_low', 'fico_range_high', 'dti', 'annual_inc', 'emp_length',
    'home_ownership', 'verification_status', 'purpose', 'addr_state',
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 
    'open_acc', 'total_acc', 'mths_since_last_delinq', 'mths_since_last_record',
    'pub_rec_bankruptcies', 'tax_liens', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_rev_accts', 'num_tl_120dpd_2m', 'pct_tl_nvr_dlq',
    'bc_open_to_buy', 'bc_util', 'all_util', 'total_rev_hi_lim',
    'inq_last_12m', 'open_act_il', 'total_bal_il', 'il_util',
    'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mths_since_recent_bc',
    'num_bc_tl', 'num_actv_rev_tl', 'num_op_rev_tl', 'num_sats',
    'application_type', 'issue_year', 'initial_list_status'
]

for col in feature_cols:
    if col in train.columns:
        null_count = train[col].isnull().sum()
        if null_count > 0:
            print(f"{col}: {null_count} nulls ({null_count/len(train)*100:.2f}%)")
