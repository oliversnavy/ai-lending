import pickle
import sys
import pandas as pd
import numpy as np

class SensitivityModel:
    pass
sys.modules['__main__'].SensitivityModel = SensitivityModel

with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("alpha_0:", model.alpha_0)
print("beta_burden:", model.beta_burden)
print("beta_match:", model.beta_match)
print("beta_spread:", model.beta_spread)
print("min_viable_rate:", model.min_viable_rate)
print("noise_std:", model.noise_std)
print("market_rates shape:", model.market_rates.shape if hasattr(model.market_rates, 'shape') else len(model.market_rates))

# Test predict_proba_batch
test_df = pd.DataFrame({
    'grade': ['C', 'D', 'E'],
    'offered_rate': [0.25, 0.28, 0.32],
    'loan_amnt': [10000, 15000, 20000],
    'annual_inc': [50000, 60000, 70000],
    'funded_amnt': [10000, 15000, 20000]
})
probs = model.predict_proba_batch(test_df)
print("\nTest predictions:")
for i, row in test_df.iterrows():
    print(f"  Grade {row['grade']} @ {row['offered_rate']:.0%}: p_accept = {probs[i]:.4f}")