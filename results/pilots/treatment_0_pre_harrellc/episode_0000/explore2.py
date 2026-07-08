
import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

print("Grade distribution:")
print(train['grade'].value_counts())
print("\nEvent rates by grade:")
for g in sorted(train['grade'].unique()):
    subset = train[train['grade'] == g]
    event_rate = subset['event'].mean()
    print(f"  {g}: {event_rate:.4f} ({len(subset)} loans)")

print("\nTerm distribution:")
print(train['term'].value_counts())

print("\nRate range:")
print(f"  Min: {train['int_rate'].min()}, Max: {train['int_rate'].max()}")

print("\nLoan amount range:")
print(f"  Min: {train['loan_amnt'].min()}, Max: {train['loan_amnt'].max()}, Mean: {train['loan_amnt'].mean():.0f}")

print("\nObserved time range:")
print(f"  Min: {train['observed_time'].min()}, Max: {train['observed_time'].max()}, Median: {train['observed_time'].median():.1f}")

print("\nEvent distribution:")
print(f"  Event=1 (charged off): {train['event'].mean():.4f} ({train['event'].sum()} events)")
