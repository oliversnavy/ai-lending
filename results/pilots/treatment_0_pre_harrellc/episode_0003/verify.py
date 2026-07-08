import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import json

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

for df in [train, val]:
    df['term_num'] = (df['term'] == '60 months').astype(int)
    df['grade_num'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

feature_cols = [
    'loan_amnt', 'funded_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util',
    'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'total_acc', 'num_tl_120dpd_2m', 'num_tl_30dpd',
    'fico_range_low', 'fico_range_high', 'term_num', 'grade_num'
]

risk_model = pickle.load(open('risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

X_val = val[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_val[col] = X_val[col].fillna(med)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
val['default_prob'] = risk_model.predict_proba(scaler.transform(X_val))[:, 1]

val_viable = val[val['grade'].isin(['C', 'D', 'E', 'F', 'G'])].copy()

base_rates = {'C': 0.27, 'D': 0.30, 'E': 0.33, 'F': 0.34, 'G': 0.34}
val_viable['base_rate'] = val_viable['grade'].map(base_rates)
val_viable['risk_adj'] = 0.4 * val_viable['default_prob']
val_viable['offered_rate'] = np.clip(val_viable['base_rate'] + val_viable['risk_adj'], 0.21, 0.36)

offer_df = val_viable[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
p_accept = sens_model.predict_proba_batch(offer_df)
val_viable['p_accept'] = p_accept

val_viable['expected_principal'] = val_viable['p_accept'] * val_viable['loan_amnt']
val_viable['expected_interest'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['offered_rate'] * (val_viable['observed_time'] / 12.0)
val_viable['expected_loss'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['default_prob']
val_viable['expected_pnl'] = val_viable['expected_interest'] - val_viable['expected_loss']
val_viable['pnl_per_dollar'] = val_viable['expected_pnl'] / val_viable['expected_principal'].replace(0, np.nan)
val_viable = val_viable.sort_values('pnl_per_dollar', ascending=False)

selected = []
cumulative_principal = 0
CAPITAL_CAP = 15_000_000

for _, row in val_viable.iterrows():
    if row['expected_principal'] <= 0:
        continue
    if cumulative_principal + row['expected_principal'] <= CAPITAL_CAP:
        selected.append(row)
        cumulative_principal += row['expected_principal']
    if cumulative_principal >= CAPITAL_CAP:
        break

selected_df = pd.DataFrame(selected)

print("=== Selected Portfolio Summary ===")
print(f"Loans selected: {len(selected_df)}")
print(f"Total expected principal: ${selected_df['expected_principal'].sum():,.0f}")
print(f"Total expected P&L: ${selected_df['expected_pnl'].sum():,.0f}")
print(f"Total expected interest: ${selected_df['expected_interest'].sum():,.0f}")
print(f"Total expected loss: ${selected_df['expected_loss'].sum():,.0f}")
print(f"Expected acceptances: {selected_df['p_accept'].sum():.0f}")
print(f"Mean offered rate: {selected_df['offered_rate'].mean():.4f}")
print(f"Mean default prob: {selected_df['default_prob'].mean():.4f}")
print(f"Mean observed time: {selected_df['observed_time'].mean():.1f} months")
print(f"Mean loan amount: {selected_df['loan_amnt'].mean():.0f}")
print(f"Mean acceptance rate: {selected_df['p_accept'].mean():.4f}")

print(f"\nP&L breakdown:")
print(f"  Interest income: ${selected_df['expected_interest'].sum():,.0f}")
print(f"  Expected losses: ${selected_df['expected_loss'].sum():,.0f}")
print(f"  Net P&L: ${selected_df['expected_pnl'].sum():,.0f}")

print(f"\nGrade distribution:")
print(selected_df['grade'].value_counts())
