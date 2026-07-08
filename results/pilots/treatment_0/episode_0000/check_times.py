import pandas as pd
import numpy as np

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check observed_time by term
print("Observed time by term:")
for term in val['term'].unique():
    sub = val[val['term'] == term]
    print(f"  {term}: mean={sub['observed_time'].mean():.1f}, median={sub['observed_time'].median():.1f}, n={len(sub)}")

# Check by grade and term
print("\nObserved time by grade and term:")
for grade in ['C', 'D', 'E', 'F', 'G']:
    for term in ['36 months', '60 months']:
        sub = val[(val['grade'] == grade) & (val['term'] == term)]
        if len(sub) > 0:
            print(f"  {grade} {term}: mean={sub['observed_time'].mean():.1f}, n={len(sub)}")

# Check actual P&L for a sample
print("\nActual P&L for a sample of Grade F loans:")
sample = val[val['grade'] == 'F'].head(1000)
for _, row in sample.iterrows():
    interest = row['loan_amnt'] * (0.35 * row['observed_time'] / 12)
    principal_lost = row['loan_amnt'] if row['event'] == 1 else 0
    pnl = interest - principal_lost
    print(f"  loan={row['loan_amnt']}, time={row['observed_time']:.1f}, event={row['event']}, interest={interest:.0f}, pnl={pnl:.0f}")
