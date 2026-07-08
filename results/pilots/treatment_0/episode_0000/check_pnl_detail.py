
import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel

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

model = SensitivityModel()

# Best strategy
def rate_fn(grade, dp):
    base = {'C': 0.35, 'D': 0.40, 'E': 0.40, 'F': 0.45, 'G': 0.45}.get(grade, 0.21)
    base += dp * 0.05
    return min(max(base, 0.21), 0.50)

test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'event', 'observed_time']].copy()
test_df['default_prob'] = val_probs
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

# Check a few examples
print("Top 5 selected loans:")
print(selected[['grade', 'loan_amnt', 'offered_rate', 'observed_time', 'event', 
                 'acceptance_prob', 'interest_collected', 'principal_lost', 'actual_pnl',
                 'expected_pnl', 'expected_principal']].head(10).to_string())

print(f"\nSummary statistics of selected loans:")
print(f"  Mean offered_rate: {selected['offered_rate'].mean():.2%}")
print(f"  Mean observed_time: {selected['observed_time'].mean():.1f} months")
print(f"  Mean event rate: {selected['event'].mean():.1%}")
print(f"  Mean acceptance_prob: {selected['acceptance_prob'].mean():.1%}")
print(f"  Mean actual_pnl: ${selected['actual_pnl'].mean():,.0f}")
print(f"  Mean expected_pnl: ${selected['expected_pnl'].mean():,.0f}")

# Check total interest and losses
total_interest = selected['interest_collected'].sum()
total_loss = selected['principal_lost'].sum()
total_expected_pnl = selected['expected_pnl'].sum()

print(f"\nTotal interest collected: ${total_interest:,.0f}")
print(f"Total principal lost: ${total_loss:,.0f}")
print(f"Total expected P&L: ${total_expected_pnl:,.0f}")

# Also check the total expected principal
print(f"Total expected principal: ${selected['expected_principal'].sum():,.0f}")
