
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel
from lifelines.utils import concordance_index
import pickle

# Full pipeline
val = pd.read_pickle('val_scored.pkl')

# Get acceptance probabilities
val['p_accept'] = pd.read_pickle('val_scored.pkl')['p_accept']  # already computed

# Recompute P&L
val['time_fraction'] = val['observed_time'] / 12.0
val['expected_principal'] = val['p_accept'] * val['loan_amnt']
val['expected_interest'] = val['p_accept'] * val['loan_amnt'] * val['offered_rate'] * val['time_fraction']
val['expected_loss'] = val['p_accept'] * val['loan_amnt'] * val['event']
val['expected_pnl'] = val['expected_interest'] - val['expected_loss']

# Filter valid
valid = val[val['p_accept'].notna() & (val['p_accept'] > 0)].copy()

# ROI
valid['roi'] = valid['expected_pnl'] / valid['expected_principal'].clip(lower=1)

# Select top by ROI under cap
cap = 15_000_000
valid_sorted = valid.sort_values('roi', ascending=False)

selected_indices = []
cumulative = 0
for i in range(len(valid_sorted)):
    row = valid_sorted.iloc[i]
    if cumulative + row['expected_principal'] <= cap:
        selected_indices.append(i)
        cumulative += row['expected_principal']
    else:
        break

selected = valid_sorted.iloc[selected_indices]

# Metrics
total_principal = selected['expected_principal'].sum()
total_pnl = selected['expected_pnl'].sum()
loans_funded = selected['p_accept'].sum()
acceptance_rate = selected['p_accept'].mean()

print(f"Total principal: ${total_principal:,.0f}")
print(f"Total P&L: ${total_pnl:,.0f}")
print(f"Loans funded: {loans_funded:.0f}")
print(f"Acceptance rate: {acceptance_rate:.4f}")
print(f"Number of loans selected: {len(selected):,}")

# Check grade distribution of selected
print(f"\nGrade distribution of selected loans:")
print(selected['grade'].value_counts())

# Check rate distribution
print(f"\nOffered rate stats:")
print(selected['offered_rate'].describe())

# Check avg time
print(f"\nAvg observed time: {selected['observed_time'].mean():.1f} months")

# Verify P&L per loan
print(f"\nP&L per funded loan: ${total_pnl / loans_funded:,.0f}")
print(f"Annualized ROI: {total_pnl / total_principal / (selected['observed_time'].mean() / 12):.2%}")
