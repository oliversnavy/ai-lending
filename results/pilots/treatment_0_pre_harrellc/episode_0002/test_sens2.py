import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Test predict_proba_batch
df = pd.DataFrame({
    'grade': ['C', 'D', 'E', 'F', 'A', 'B', 'C', 'D'],
    'offered_rate': [0.24, 0.28, 0.30, 0.36, 0.21, 0.21, 0.24, 0.28],
    'loan_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    'annual_inc': [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000],
    'funded_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
})

probs = model.predict_proba_batch(df)
print("Acceptance probabilities:")
for i, row in df.iterrows():
    print(f"  {row['grade']} @ {row['offered_rate']:.0%}: {probs[i]:.4f}")

# Also check the model's internal attributes
print("\nModel info:")
print(f"  min_viable_rate: {model.min_viable_rate}")
print(f"  market_rates: {model.market_rates}")
print(f"  noise_std: {model.noise_std}")
print(f"  beta_spread: {model.beta_spread}")
print(f"  beta_burden: {model.beta_burden}")
print(f"  beta_match: {model.beta_match}")
print(f"  alpha_0: {model.alpha_0}")
