import pandas as pd
import numpy as np
import pickle

# Explore validation data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
print("Val shape:", val.shape)
print("\nVal grade distribution:")
print(val['grade'].value_counts())
print("\nVal event rate by grade:")
print(val.groupby('grade')['event'].agg(['count', 'mean']))

# Check sensitivity model
model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))
print("\nSensitivity model type:", type(model))
print("Sensitivity model features:", model.feature_names if hasattr(model, 'feature_names') else "N/A")

# Quick test of sensitivity model
test_df = pd.DataFrame({
    'grade': ['C', 'D', 'E'],
    'offered_rate': [0.25, 0.28, 0.32],
    'loan_amnt': [10000, 15000, 20000],
    'annual_inc': [50000, 60000, 70000],
    'funded_amnt': [10000, 15000, 20000]
})
probs = model.predict_proba_batch(test_df)
print("\nSensitivity model test:")
for i, row in test_df.iterrows():
    print(f"  Grade {row['grade']} @ {row['offered_rate']:.0%}: p_accept = {probs[i]:.4f}")