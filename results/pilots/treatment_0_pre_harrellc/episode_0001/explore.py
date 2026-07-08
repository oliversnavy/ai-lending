import pandas as pd
import numpy as np

# Load and explore training data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("Train shape:", train.shape)
print("\nColumns:", train.columns.tolist())
print("\n--- Key columns summary ---")
print(train[['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'sub_grade',
             'fico_range_low', 'fico_range_high', 'dti', 'annual_inc', 'event', 'observed_time']].describe())
print("\n--- Grade distribution ---")
print(train['grade'].value_counts())
print("\n--- Event rate by grade ---")
print(train.groupby('grade')['event'].mean())
print("\n--- Event rates by grade (count) ---")
print(train.groupby('grade')['event'].agg(['count', 'mean']))