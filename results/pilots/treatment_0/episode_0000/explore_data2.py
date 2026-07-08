
import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Explore grade distribution and event rates
print("Grade distribution in train:")
print(train['grade'].value_counts())

print("\nEvent rate by grade:")
for g in sorted(train['grade'].unique()):
    subset = train[train['grade'] == g]
    event_rate = subset['event'].mean()
    print(f"  Grade {g}: {len(subset)} loans, event rate = {event_rate:.4f}")

print("\nEvent rate by sub_grade:")
for g in sorted(train['sub_grade'].unique()):
    subset = train[train['sub_grade'] == g]
    event_rate = subset['event'].mean()
    print(f"  {g}: {len(subset)} loans, event rate = {event_rate:.4f}")

print("\nKey stats:")
print(f"  Mean loan_amnt: {train['loan_amnt'].mean():.0f}")
print(f"  Mean int_rate: {train['int_rate'].mean():.2f}%")
print(f"  Mean annual_inc: {train['annual_inc'].mean():.0f}")
print(f"  Mean dti: {train['dti'].mean():.1f}")
print(f"  Event rate overall: {train['event'].mean():.4f}")
print(f"  Mean observed_time: {train['observed_time'].mean():.1f} months")

print("\nTerm distribution:")
print(train['term'].value_counts())

print("\nFICO range stats:")
print(f"  fico_range_low: min={train['fico_range_low'].min()}, max={train['fico_range_low'].max()}, mean={train['fico_range_low'].mean():.0f}")
print(f"  fico_range_high: min={train['fico_range_high'].min()}, max={train['fico_range_high'].max()}, mean={train['fico_range_high'].mean():.0f}")
