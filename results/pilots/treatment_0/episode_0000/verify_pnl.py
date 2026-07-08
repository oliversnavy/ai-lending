# Let me verify the P&L calculation and improve the model
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')
from data_pipeline.sensitivity_model import SensitivityModel
import pickle
from sklearn.metrics import roc_auc_score

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check actual default rates
print("Actual default rates in val:")
for g in ['C', 'D', 'E', 'F', 'G']:
    sub = val[val['grade'] == g]
    print(f"  {g}: {sub['event'].mean():.4f} (n={len(sub)})")

# Check observed_time distribution
print(f"\nObserved time stats:")
print(val['observed_time'].describe())

# The key insight: observed_time varies significantly
# For 36-month loans, observed_time might be ~30 months (early payoff)
# For 60-month loans, observed_time might be ~50 months

# Let me check by term
for term in ['36 months', '60 months']:
    sub = val[val['term'] == term]
    print(f"\n{term}:")
    print(f"  observed_time: mean={sub['observed_time'].mean():.1f}, median={sub['observed_time'].median():.1f}")
