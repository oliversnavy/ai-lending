
import pandas as pd
import numpy as np
import pickle

# Load validation data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
print("Val shape:", val.shape)
print("\nGrade distribution in val:")
print(val['grade'].value_counts())
print("\nEvent rates by grade in val:")
for g in sorted(val['grade'].unique()):
    subset = val[val['grade'] == g]
    print(f"  {g}: {subset['event'].mean():.4f} ({len(subset)} loans)")
