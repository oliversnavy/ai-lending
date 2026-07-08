
import pandas as pd
import numpy as np

# Load training data using absolute path
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

print("Train shape:", train.shape)
print("Val shape:", val.shape)

print("\nTrain columns:", list(train.columns))
print("\nTrain dtypes:")
print(train.dtypes)
