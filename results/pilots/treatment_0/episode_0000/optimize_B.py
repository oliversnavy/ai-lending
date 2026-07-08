
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

# Target: Optimize Strategy B parameters
# Base rates: C=0.30, D=0.35, E=0.35, F=0.40, G=0.40
# Risk premium: dp * 0.05

def evaluate_rate_params(base_rates, risk_mult):
    def rate_fn(grade, dp):
        base = base_rates.get(grade, 0.21)
        base += dp * risk_mult
        return min(max(base, 0.21), 0.50)
    
    test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob', 
                   'event', 'observed_time']].copy()
    test_df['offered_rate'] = test_df.apply(
        lambda row: rate_fn(row['grade'], row['default_prob']), axis=1
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
    
    return selected['expected_pnl'].sum(), selected['acceptance_prob'].sum(), selected['expected_principal'].sum()

# Grid search over base rates
best_pnl = 0
best_params = None

for bC in [0.28, 0.30, 0.32, 0.35, 0.38]:
    for bD in [0.33, 0.35, 0.38, 0.40, 0.42]:
        for bE in [0.33, 0.35, 0.38, 0.40, 0.42]:
            for bF in [0.38, 0.40, 0.42, 0.45, 0.48]:
                for bG in [0.38, 0.40, 0.42, 0.45, 0.48]:
                    for rm in [0.03, 0.05, 0.08, 0.10]:
                        base_rates = {'C': bC, 'D': bD, 'E': bE, 'F': bF, 'G': bG}
                        pnl, loans, principal = evaluate_rate_params(base_rates, rm)
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_params = (bC, bD, bE, bF, bG, rm)
                            print(f"New best: P&L=${pnl:,.0f}, params=({bC}, {bD}, {bE}, {bF}, {bG}, {rm})")

print(f"\nBest P&L: ${best_pnl:,.0f}")
print(f"Best params: base_C={best_params[0]}, base_D={best_params[1]}, base_E={best_params[2]}, "
      f"base_F={best_params[3]}, base_G={best_params[4]}, risk_mult={best_params[5]}")
