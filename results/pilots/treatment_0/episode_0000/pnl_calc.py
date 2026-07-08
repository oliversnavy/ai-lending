
import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Look at observed_time by event
print("Observed time by event:")
print(train.groupby('event')['observed_time'].describe())

print("\nObserved time by grade and event:")
for g in ['C', 'D', 'E', 'F']:
    subset = train[train['grade'] == g]
    for e in [0, 1]:
        sub = subset[subset['event'] == e]
        if len(sub) > 0:
            print(f"  Grade {g}, event={e}: mean_time={sub['observed_time'].mean():.1f} months, n={len(sub)}")

# Calculate expected P&L for a sample loan
# interest_collected = loan_amnt * (offered_rate * observed_time / 12)
# principal_lost = loan_amnt if event == 1 else 0

print("\n\nSample P&L calculation:")
loan_amnt = 15000
offered_rate = 0.21

# For Grade C (18% default rate)
# Non-defaulters pay full term (36 months)
# Defaulters pay for observed_time months
c_non_default = train[(train['grade'] == 'C') & (train['event'] == 0)]['observed_time'].mean()
c_default = train[(train['grade'] == 'C') & (train['event'] == 1)]['observed_time'].mean()
c_default_rate = train[(train['grade'] == 'C')]['event'].mean()

print(f"Grade C: non-default avg time = {c_non_default:.1f} months")
print(f"Grade C: default avg time = {c_default:.1f} months")
print(f"Grade C: default rate = {c_default_rate:.4f}")

# Expected P&L per loan
interest_nd = loan_amnt * offered_rate * c_non_default / 12
interest_d = loan_amnt * offered_rate * c_default / 12
loss_d = loan_amnt

expected_pnl = (1 - c_default_rate) * interest_nd - c_default_rate * loss_d
print(f"\nExpected P&L per Grade C loan at 21%:")
print(f"  Interest from non-defaulters: {interest_nd:.0f}")
print(f"  Interest from defaulters: {interest_d:.0f}")
print(f"  Expected interest: {(1-c_default_rate)*interest_nd + c_default_rate*interest_d:.0f}")
print(f"  Expected loss: {c_default_rate * loss_d:.0f}")
print(f"  Net expected P&L: {expected_pnl:.0f}")

# Try different rates
for rate in [0.21, 0.25, 0.30, 0.35, 0.40]:
    expected_pnl = (1 - c_default_rate) * loan_amnt * rate * c_non_default / 12 - c_default_rate * loan_amnt
    print(f"  Rate {rate:.0%}: Expected P&L = {expected_pnl:.0f}")
