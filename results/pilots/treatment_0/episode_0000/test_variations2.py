
import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel
from sklearn.metrics import roc_auc_score

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

with open('risk_model.pkl', 'rb') as f:
    gbc = pickle.load(f)

val_probs = gbc.predict_proba(val_enc.values.astype(float))[:, 1]
val['default_prob'] = val_probs

model = SensitivityModel()

# Try rates between B_higher and B_even_higher
strategies = {
    'B_higher_0.05': lambda g, dp: min(max({'C': 0.35, 'D': 0.40, 'E': 0.40, 'F': 0.45, 'G': 0.45}.get(g, 0.21) + dp * 0.05, 0.21), 0.50),
    'B_higher_0.08': lambda g, dp: min(max({'C': 0.35, 'D': 0.40, 'E': 0.40, 'F': 0.45, 'G': 0.45}.get(g, 0.21) + dp * 0.08, 0.21), 0.50),
    'B_higher_0.10': lambda g, dp: min(max({'C': 0.35, 'D': 0.40, 'E': 0.40, 'F': 0.45, 'G': 0.45}.get(g, 0.21) + dp * 0.10, 0.21), 0.50),
    'B_even_higher_0.05': lambda g, dp: min(max({'C': 0.40, 'D': 0.45, 'E': 0.45, 'F': 0.50, 'G': 0.50}.get(g, 0.21) + dp * 0.05, 0.21), 0.50),
    'B_even_higher_0.03': lambda g, dp: min(max({'C': 0.40, 'D': 0.45, 'E': 0.45, 'F': 0.50, 'G': 0.50}.get(g, 0.21) + dp * 0.03, 0.21), 0.50),
    'B_higher_0.15': lambda g, dp: min(max({'C': 0.35, 'D': 0.40, 'E': 0.40, 'F': 0.45, 'G': 0.45}.get(g, 0.21) + dp * 0.15, 0.21), 0.50),
}

for name, fn in strategies.items():
    test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob', 
                   'event', 'observed_time']].copy()
    test_df['offered_rate'] = test_df.apply(
        lambda row: fn(row['grade'], row['default_prob']), axis=1
    )
    test_df = test_df[test_df['offered_rate'] > 0]
    
    acceptance_probs = model.predict_proba_batch(test_df)
    test_df['acceptance_prob'] = acceptance_probs
    
    test_df['interest_collected'] = test_df['loan_amnt'] * test_df['offered_rate'] * test_df['observed_time'] / 12
    test_df['principal_lost'] = test_df['loan_amnt'] * test_df['event']
    test_df['actual_pnl'] = test_df['interest_collected'] - test_df['principal_lost']
    test_df['expected_pnl'] = test_df['acceptance_prob'] * test_df['actual_pnl']
    test_df['expected_principal'] = test_df['loan_amnt'] * test_df['acceptance_prob']
    
    sorted_df = test_df.sort_values('expected_pnl', ascending=False)
    cumulative = sorted_df['expected_principal'].cumsum()
    selected = sorted_df[cumulative <= 15000000]
    
    pnl = selected['expected_pnl'].sum()
    loans = selected['acceptance_prob'].sum()
    principal = selected['expected_principal'].sum()
    acc_rate = selected['acceptance_prob'].mean()
    
    print(f"{name}: P&L=${pnl:,.0f}, Loans={loans:.0f}, Principal=${principal:,.0f}, "
          f"Acceptance={acc_rate:.1%}")
