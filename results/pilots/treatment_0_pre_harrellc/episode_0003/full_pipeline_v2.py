import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

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

# Score val
X_val = val[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_val[col] = X_val[col].fillna(med)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
val['default_prob'] = risk_model.predict_proba(scaler.transform(X_val))[:, 1]

# Filter to viable segment (exclude G - negative P&L)
val_viable = val[val['grade'].isin(['C', 'D', 'E', 'F'])].copy()
print(f"Viable applicants (C-F): {len(val_viable)}")

# Optimized pricing: grade-specific base rates + risk adjustment
base_rates = {'C': 0.24, 'D': 0.28, 'E': 0.33, 'F': 0.36}
risk_adj_coeff = 0.3  # moderate risk adjustment

val_viable['base_rate'] = val_viable['grade'].map(base_rates)
val_viable['risk_adj'] = risk_adj_coeff * val_viable['default_prob']
val_viable['offered_rate'] = np.clip(val_viable['base_rate'] + val_viable['risk_adj'], 0.21, 0.36)

# Batch predict acceptance
offer_df = val_viable[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
p_accept = sens_model.predict_proba_batch(offer_df)
val_viable['p_accept'] = p_accept

# Compute P&L
val_viable['expected_principal'] = val_viable['p_accept'] * val_viable['loan_amnt']
val_viable['expected_interest'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['offered_rate'] * (val_viable['observed_time'] / 12.0)
val_viable['expected_loss'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['default_prob']
val_viable['expected_pnl'] = val_viable['expected_interest'] - val_viable['expected_loss']
val_viable['pnl_per_dollar'] = val_viable['expected_pnl'] / val_viable['expected_principal'].replace(0, np.nan)

# Sort by P&L per dollar for greedy selection
val_viable = val_viable.sort_values('pnl_per_dollar', ascending=False)

# Apply capital cap
CAPITAL_CAP = 15_000_000
selected = []
cumulative_principal = 0

for _, row in val_viable.iterrows():
    if row['expected_principal'] <= 0:
        continue
    if cumulative_principal + row['expected_principal'] <= CAPITAL_CAP:
        selected.append(row)
        cumulative_principal += row['expected_principal']
    if cumulative_principal >= CAPITAL_CAP:
        break

selected_df = pd.DataFrame(selected)

# Compute metrics
total_principal = selected_df['expected_principal'].sum()
loans_funded = selected_df['p_accept'].sum()
total_pnl = selected_df['expected_pnl'].sum()
acceptance_rate = selected_df['p_accept'].mean()

print(f"\n=== PORTFOLIO METRICS ===")
print(f"Total principal deployed: ${total_principal:,.0f}")
print(f"Expected loans funded: {loans_funded:.0f}")
print(f"Total expected P&L: ${total_pnl:,.0f}")
print(f"Acceptance rate: {acceptance_rate:.4f}")
print(f"P&L per dollar: ${total_pnl / total_principal:.4f}")
print(f"Loans selected: {len(selected_df)}")

# Check constraints
print(f"\nCapital cap met: {total_principal <= CAPITAL_CAP}")
print(f"Volume floor met: {loans_funded >= 400}")

# Compute C-index from val set
from lifelines.utils import concordance_index
ci = concordance_index(val['observed_time'], -val['default_prob'], val['event'])
print(f"C-index: {ci:.4f}")

# Write results
results_json = {
    "pnl": float(total_pnl),
    "c_stat": float(ci),
    "acceptance_rate": float(acceptance_rate),
    "loans_funded": int(round(loans_funded)),
    "total_principal": float(total_principal),
    "approach": "Logistic regression risk model (C-index 0.68) scores applicants; grade-specific optimal rates (24-36%) with risk adjustment; Grade C-F targeted; greedy capital allocation by P&L/dollar",
    "hypothesis": "Grade-specific pricing (24% for C, 28% for D, 33% for E, 36% for F) maximizes expected P&L by balancing acceptance probability against interest income and default losses"
}

with open('results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("\nresults.json written!")

# Print grade distribution
print(f"\nGrade distribution:")
print(selected_df['grade'].value_counts())
print(f"\nMean offered rate: {selected_df['offered_rate'].mean():.4f}")
print(f"Mean default prob: {selected_df['default_prob'].mean():.4f}")
