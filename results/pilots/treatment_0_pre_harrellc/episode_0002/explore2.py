import pandas as pd
import numpy as np

# Load validation data to understand its structure
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
print("Val shape:", val.shape)
print("Val grade distribution:")
print(val['grade'].value_counts())
print("\nVal event rate by grade:")
print(val.groupby('grade')['event'].mean())

# Check key columns
print("\nVal sample:")
print(val[['grade','int_rate','loan_amnt','funded_amnt','dti','annual_inc','event','observed_time']].head(5))

# Check term distribution
print("\nTerm distribution:")
print(val['term'].value_counts())
