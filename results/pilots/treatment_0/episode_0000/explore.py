import pandas as pd
import numpy as np

# Load training data with absolute path
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("Train shape:", train.shape)
print("\nColumns:", train.columns.tolist())
print("\nFirst few rows:")
print(train.head(3))
