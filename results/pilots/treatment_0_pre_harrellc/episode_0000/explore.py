
import pandas as pd
import numpy as np

# Load training data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("Train shape:", train.shape)
print("\nColumns:", list(train.columns))
print("\nFirst few rows:")
print(train.head())
