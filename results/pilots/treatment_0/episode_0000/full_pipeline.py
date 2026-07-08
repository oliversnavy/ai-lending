
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

def get_offered_rate(grade, default_prob):
    base_rates = {
        'C': 0.21,
        'D': 0.25,
        'E': 0.21,
        'F': 0.35,
        'G': 0.30,
    }
    base = base_rates.get(grade, 0.21)
    
    if default_prob > 0.35:
        base += 0.05
    elif default_prob > 0.25:
        base += 0.03
    
    return min(base, 0.50)

# Create full test dataframe
test_df = val[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt', 'default_prob', 
               'event', 'observed_time', 'term']].copy()
test_df['offered_rate'] = test_df.apply(
    lambda row: get_offered_rate(row['grade'], row['default_prob']), axis=1
)

# Calculate acceptance probabilities
acceptance_probs = model.predict_proba_batch(test_df)
test_df['acceptance_prob'] = acceptance_probs

# Filter to Grade C-F
viable = test_df[test_df['grade'].isin(['C', 'D', 'E', 'F'])].copy()

# Calculate expected P&L
viable['expected_interest'] = viable['loan_amnt'] * viable['offered_rate'] * viable['observed_time'] / 12
viable['expected_loss'] = viable['loan_amnt'] * viable['default_prob']
viable['expected_pnl'] = viable['acceptance_prob'] * (viable['expected_interest'] - viable['expected_loss'])
viable['pnl_per_dollar'] = viable['expected_pnl'] / viable['loan_amnt']
viable['expected_principal'] = viable['loan_amnt'] * viable['acceptance_prob']

# Sort by expected P&L per dollar
viable_sorted = viable.sort_values('pnl_per_dollar', ascending=False)

# Apply capital cap: $15M on expected principal
cumulative_principal = viable_sorted['expected_principal'].cumsum()
cap_mask = cumulative_principal <= 15000000
selected = viable_sorted[cap_mask]

# Check if we meet volume floor
expected_loans = selected['acceptance_prob'].sum()
print(f"Expected loans funded: {expected_loans:.0f}")
print(f"Total expected principal: ${selected['expected_principal'].sum():,.0f}")
print(f"Total P&L: ${selected['expected_pnl'].sum():,.0f}")

# If we don't meet the 400 loan floor, we need to expand
if expected_loans < 400:
    print("\nWARNING: Not meeting volume floor of 400 loans!")
    # Expand to include more loans until we hit 400
    remaining = viable_sorted[~cap_mask]
    while expected_loans < 400 and len(remaining) > 0:
        chunk = remaining[:1000]  # Add in chunks
        selected = pd.concat([selected, chunk])
        expected_loans = selected['acceptance_prob'].sum()
        cumulative_principal = selected['expected_principal'].sum()
        remaining = remaining[1000:]
        print(f"  Added chunk, now {expected_loans:.0f} expected loans, ${cumulative_principal:,.0f} principal")

print(f"\nFinal: {len(selected)} applicants, {expected_loans:.0f} expected loans, "
      f"${selected['expected_principal'].sum():,.0f} principal, "
      f"${selected['expected_pnl'].sum():,.0f} P&L")
