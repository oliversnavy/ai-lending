import pandas as pd
import numpy as np

# Load training data with absolute path
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("Train shape:", train.shape)
print("\nColumns:", list(train.columns))
print("\nFirst 5 rows:")
print(train.head())
print("\nBasic stats:")
print(train[['loan_amnt', 'funded_amnt', 'int_rate', 'annual_inc', 'dti']].describe())