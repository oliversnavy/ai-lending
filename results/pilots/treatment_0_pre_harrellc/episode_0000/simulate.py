
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel, _sigmoid

# Load scored validation data
val = pd.read_pickle('val_scored.pkl')

# Create sensitivity model
sens_model = SensitivityModel()

# Prepare batch for sensitivity model
batch = val[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()

# Get acceptance probabilities
p_accept = sens_model.predict_proba_batch(batch)

val['p_accept'] = p_accept

print(f"Mean acceptance rate: {p_accept.mean():.4f}")
print(f"Acceptance rate by grade:")
for g in ['C', 'D', 'E', 'F']:
    subset = val[val['grade'] == g]
    if len(subset) > 0:
        print(f"  {g}: {subset['p_accept'].mean():.4f} (n={len(subset):,})")

# Now compute P&L
# For each loan:
#   expected_principal = p_accept * loan_amnt
#   expected_interest = p_accept * loan_amnt * offered_rate * (observed_time / 12)
#   expected_loss = p_accept * loan_amnt * event (event=1 if charged off)
#   expected_pnl = expected_interest - expected_loss

val['time_fraction'] = val['observed_time'] / 12.0

val['expected_principal'] = val['p_accept'] * val['loan_amnt']
val['expected_interest'] = val['p_accept'] * val['loan_amnt'] * val['offered_rate'] * val['time_fraction']
val['expected_loss'] = val['p_accept'] * val['loan_amnt'] * val['event']
val['expected_pnl'] = val['expected_interest'] - val['expected_loss']

# Filter to loans within capital cap
total_principal = val['expected_principal'].sum()
total_pnl = val['expected_pnl'].sum()
loans_funded = val['p_accept'].sum()

print(f"\n--- Initial Portfolio Stats ---")
print(f"Total expected principal: ${total_principal:,.0f}")
print(f"Total expected P&L: ${total_pnl:,.0f}")
print(f"Expected loans funded: {loans_funded:.0f}")
print(f"Acceptance rate: {val['p_accept'].mean():.4f}")

# Check if we exceed capital cap
if total_principal > 15_000_000:
    print(f"\nExceeds capital cap! Need to rank and select top {15_000_000 / total_principal:.1%} by ROI")
    # Rank by expected_pnl / expected_principal
    val['roi'] = val['expected_pnl'] / val['expected_principal'].clip(lower=1)
    val_sorted = val.sort_values('roi', ascending=False)
    
    cumulative_principal = 0
    selected = []
    for _, row in val_sorted.iterrows():
        if cumulative_principal + row['expected_principal'] <= 15_000_000:
            selected.append(row)
            cumulative_principal += row['expected_principal']
        else:
            # Partial fill
            remaining = 15_000_000 - cumulative_principal
            if remaining > 0:
                frac = remaining / row['expected_principal']
                selected.append({
                    'expected_principal': row['expected_principal'] * frac,
                    'expected_pnl': row['expected_pnl'] * frac,
                    'p_accept': row['p_accept'] * frac
                })
                cumulative_principal += row['expected_principal'] * frac
            break
    
    if len(selected) > 0:
        sel_df = pd.DataFrame(selected)
        total_principal = sel_df['expected_principal'].sum()
        total_pnl = sel_df['expected_pnl'].sum()
        loans_funded = sel_df['p_accept'].sum()
        print(f"After cap: principal={total_principal:,.0f}, pnl={total_pnl:,.0f}, funded={loans_funded:.0f}")
else:
    print(f"\nWithin capital cap.")
