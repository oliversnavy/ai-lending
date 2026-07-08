import pickle
import pandas as pd

# Load sensitivity model
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)

print("Sensitivity model type:", type(sens_model))
print("Sensitivity model dir:", [x for x in dir(sens_model) if not x.startswith('_')])

# Test with a sample
df_test = pd.DataFrame({
    'grade': ['C'],
    'offered_rate': [0.25],
    'loan_amnt': [10000],
    'annual_inc': [50000],
    'funded_amnt': [8000]
})
probs = sens_model.predict_proba_batch(df_test)
print("\nTest acceptance probability:", probs)