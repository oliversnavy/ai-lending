import numpy as np
import pandas as pd

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check observed_time distribution
print("Observed time stats:")
print(val['observed_time'].describe())
print(f"\nMedian observed time: {val['observed_time'].median():.1f} months")
print(f"Mean observed time: {val['observed_time'].mean():.1f} months")

# Check event rate
print(f"\nEvent rate: {val['event'].mean():.2%}")

# Check by grade
print("\nEvent rate by grade:")
print(val.groupby('grade')['event'].mean())

# Check average loan_amnt
print(f"\nAverage loan_amnt: ${val['loan_amnt'].mean():,.0f}")
print(f"Median loan_amnt: ${val['loan_amnt'].median():,.0f}")

# Check average annual_inc
print(f"\nAverage annual_inc: ${val['annual_inc'].mean():,.0f}")

# Check grade distribution
print("\nGrade distribution:")
print(val['grade'].value_counts(normalize=True))

# Check term distribution
print("\nTerm distribution:")
print(val['term'].value_counts(normalize=True))

# Check average interest rate in market
print("\nMarket interest rates by grade:")
print(val.groupby('grade')['int_rate'].mean())