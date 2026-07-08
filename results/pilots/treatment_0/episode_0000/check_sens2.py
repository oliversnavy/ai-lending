
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Quick check on sensitivity model
test_df = pd.DataFrame({
    'grade': ['C', 'D', 'E', 'A', 'B'],
    'offered_rate': [0.25, 0.28, 0.30, 0.21, 0.21],
    'loan_amnt': [10000, 15000, 20000, 10000, 10000],
    'annual_inc': [50000, 60000, 70000, 80000, 80000],
    'funded_amnt': [10000, 15000, 20000, 10000, 10000]
})
probs = SensitivityModel().predict_proba_batch(test_df)
print("Acceptance probs:", probs)
print("\nNote: A and B should have near-zero acceptance at 21%")
