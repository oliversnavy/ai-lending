
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

# Grid search over rate parameters
best_pnl = 0
best_params = None

for base_C in [0.25, 0.28, 0.30, 0.32, 0.35]:
    for base_D in [0.30, 0.33, 0.35, 0.38, 0.40]:
        for base_E in [0.30, 0.33, 0.35, 0.38, 0.40]:
            for base_F in [0.35, 0.38, 0.40, 0.42, 0.45]:
                for base_G in [0.35, 0.38, 0.40, 0.42, 0.45]:
                    for risk_mult in [0.05, 0.08, 0.10, 0.12, 0.15]:
                        
                        def rate_fn(grade, dp):
                            base = {'C': base_C, 'D': base_D, 'E': base_E, 'F': base_F, 'G': base_G}.get(grade, 0.21)
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
                        
                        test_df['expected_interest'] = test_df['loan_amnt'] * test_df['offered_rate'] * test_df['observed_time'] / 12
                        test_df['expected_loss'] = test_df['loan_amnt'] * test_df['default_prob']
                        test_df['expected_pnl'] = test_df['acceptance_prob'] * (test_df['expected_interest'] - test_df['expected_loss'])
                        test_df['expected_principal'] = test_df['loan_amnt'] * test_df['acceptance_prob']
                        
                        sorted_df = test_df.sort_values('expected_pnl', ascending=False)
                        cumulative = sorted_df['expected_principal'].cumsum()
                        selected = sorted_df[cumulative <= 15000000]
                        
                        pnl = selected['expected_pnl'].sum()
                        
                        if pnl > best_pnl:
                            best_pnl = pnl
                            best_params = {
                                'base_C': base_C, 'base_D': base_D, 'base_E': base_E,
                                'base_F': base_F, 'base_G': base_G, 'risk_mult': risk_mult,
                                'pnl': pnl
                            }
                            print(f"New best: P&L=${pnl:,.0f}, params={best_params}")

print(f"\nBest P&L: ${best_pnl:,.0f}")
print(f"Best params: {best_params}")
