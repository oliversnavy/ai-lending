
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel

# Load scored validation data
val = pd.read_pickle('val_scored.pkl')

# Create sensitivity model
sens_model = SensitivityModel()

# Get acceptance probabilities
batch = val[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
p_accept = sens_model.predict_proba_batch(batch)
val['p_accept'] = p_accept

# Compute P&L components
val['time_fraction'] = val['observed_time'] / 12.0
val['expected_principal'] = val['p_accept'] * val['loan_amnt']
val['expected_interest'] = val['p_accept'] * val['loan_amnt'] * val['offered_rate'] * val['time_fraction']
val['expected_loss'] = val['p_accept'] * val['loan_amnt'] * val['event']
val['expected_pnl'] = val['expected_interest'] - val['expected_loss']

# Only consider loans where we made an offer (not NaN)
valid_mask = val['p_accept'].notna() & (val['p_accept'] > 0)
val_valid = val[valid_mask].copy()

print(f"Valid loans with offers: {len(val_valid):,}")
print(f"Mean acceptance rate: {val_valid['p_accept'].mean():.4f}")

# Compute total expected principal
total_principal = val_valid['expected_principal'].sum()
print(f"Total expected principal: ${total_principal:,.0f}")

# ROI per loan
val_valid['roi'] = val_valid['expected_pnl'] / val_valid['expected_principal'].clip(lower=1)

# Rank by ROI descending
val_sorted = val_valid.sort_values('roi', ascending=False)

# Greedy selection under capital cap
cap = 15_000_000
cumulative = 0
selected_idx = []
for i in range(len(val_sorted)):
    row = val_sorted.iloc[i]
    if cumulative + row['expected_principal'] <= cap:
        selected_idx.append(i)
        cumulative += row['expected_principal']
    else:
        break

selected = val_sorted.iloc[selected_idx]
total_principal = selected['expected_principal'].sum()
total_pnl = selected['expected_pnl'].sum()
loans_funded = selected['p_accept'].sum()

print(f"\n--- After Greedy Selection ---")
print(f"Loans selected: {len(selected):,}")
print(f"Total expected principal: ${total_principal:,.0f}")
print(f"Total expected P&L: ${total_pnl:,.0f}")
print(f"Expected loans funded: {loans_funded:.0f}")
print(f"Acceptance rate (selected): {selected['p_accept'].mean():.4f}")
print(f"ROI: {total_pnl / total_principal:.4f}")
