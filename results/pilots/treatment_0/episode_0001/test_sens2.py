import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending/data/processed')

class SensitivityModel:
    pass

import pickle
import pandas as pd
import numpy as np

with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sensitivity_model = pickle.load(f)

# Test batch
test_df = pd.DataFrame({
    'grade': ['C', 'C', 'D', 'D', 'E', 'E', 'B', 'B', 'A', 'A'],
    'offered_rate': [0.21, 0.25, 0.21, 0.25, 0.21, 0.25, 0.21, 0.14, 0.21, 0.14],
    'loan_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    'annual_inc': [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000],
    'funded_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
})

probs = sensitivity_model.predict_proba_batch(test_df)
for i, row in test_df.iterrows():
    print(f"Grade {row['grade']}, rate {row['offered_rate']:.0%}: acceptance prob = {probs[i]:.4f}")

# Also check the model internals
print(f"\nMin viable rate: {sensitivity_model.min_viable_rate}")
print(f"Market rates: {sensitivity_model.market_rates}")
print(f"alpha_0: {sensitivity_model.alpha_0}")
print(f"beta_spread: {sensitivity_model.beta_spread}")