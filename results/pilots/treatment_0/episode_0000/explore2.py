import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Grade distribution
print("Grade distribution:")
print(train['grade'].value_counts())

# Event rates by grade
print("\nEvent (default) rates by grade:")
for g in sorted(train['grade'].unique()):
    subset = train[train['grade'] == g]
    evt_rate = subset['event'].mean()
    print(f"  {g}: {evt_rate:.4f} (n={len(subset)})")

# Key stats
print("\nKey stats:")
print(f"Event rate overall: {train['event'].mean():.4f}")
print(f"Median loan_amnt: {train['loan_amnt'].median()}")
print(f"Median annual_inc: {train['annual_inc'].median()}")
print(f"Median dti: {train['dti'].median()}")
print(f"Median int_rate: {train['int_rate'].median()}")
print(f"Median funded_amnt: {train['funded_amnt'].median()}")
print(f"Median observed_time: {train['observed_time'].median()}")

# Missing values
print("\nMissing values (top 20):")
print(train.isnull().sum().sort_values(ascending=False).head(20))
