import pandas as pd
import numpy as np

# Load training data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

print("Train shape:", train.shape)
print("Val shape:", val.shape)
print("\nTrain columns:", list(train.columns))
print("\nGrade distribution:")
print(train['grade'].value_counts())
print("\nEvent distribution:")
print(train['event'].value_counts())
print("\nEvent rate by grade:")
print(train.groupby('grade')['event'].mean())
print("\nInt rate stats:")
print(train['int_rate'].describe())
print("\nTerm distribution:")
print(train['term'].value_counts())
