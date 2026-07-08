import pandas as pd
import numpy as np

# Load training data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("Train shape:", train.shape)
print("\nColumns:", train.columns.tolist())
print("\nGrade distribution:")
print(train['grade'].value_counts())
print("\nEvent rate by grade:")
print(train.groupby('grade')['event'].mean())
print("\nTarget stats:")
print("event:", train['event'].mean())
print("observed_time:", train['observed_time'].describe())
print("\nSample dtypes for key columns:")
print(train[['loan_amnt','funded_amnt','term','int_rate','grade','dti','annual_inc','event','observed_time']].dtypes)
print("\nSample rows:")
print(train[['grade','int_rate','loan_amnt','funded_amnt','dti','annual_inc','event','observed_time']].head(10))
