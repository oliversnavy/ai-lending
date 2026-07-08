import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Check for columns that are too sparse or constant
cols = ['loan_amnt', 'funded_amnt', 'term_months', 'int_rate', 'grade_enc', 'sub_grade_enc',
        'fico_range_low', 'fico_range_high', 'dti', 'annual_inc', 'emp_length_enc',
        'home_ownership_enc', 'verification_status_enc', 'purpose_enc', 'state_code',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 
        'open_acc', 'total_acc', 'mths_since_last_delinq', 'mths_since_last_record',
        'pub_rec_bankruptcies', 'tax_liens', 'collections_12_mths_ex_med',
        'acc_now_delinq', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'num_rev_accts', 'num_tl_120dpd_2m', 'pct_tl_nvr_dlq',
        'bc_open_to_buy', 'bc_util', 'all_util', 'total_rev_hi_lim',
        'inq_last_12m', 'open_act_il', 'total_bal_il', 'il_util',
        'mo_sin_rcnt_tl', 'mo_sin_rcnt_rev_tl_op', 'mths_since_recent_bc',
        'num_bc_tl', 'num_actv_rev_tl', 'num_op_rev_tl', 'num_sats',
        'application_type_enc', 'issue_year', 'initial_list_status_enc']

for c in cols:
    if c in train.columns:
        null_pct = train[c].isnull().mean()
        unique = train[c].nunique()
        print(f"{c}: {null_pct:.1%} null, {unique} unique")
