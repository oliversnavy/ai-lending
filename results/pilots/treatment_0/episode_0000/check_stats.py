
import pandas as pd
import numpy as np
import pickle
from data_pipeline.sensitivity_model import SensitivityModel
from sklearn.metrics import roc_auc_score

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check average observed time and default rate by grade
for df_name, df in [('train', train), ('val', val)]:
    print(f"\n=== {df_name} ===")
    for grade in ['C', 'D', 'E', 'F', 'G']:
        subset = df[df['grade'] == grade]
        avg_time = subset['observed_time'].mean()
        default_rate = subset['event'].mean()
        print(f"  {grade}: avg_time={avg_time:.1f} months, default_rate={default_rate:.1%}")

# Also check the distribution of observed_time
print(f"\nOverall observed_time stats:")
print(val['observed_time'].describe())
