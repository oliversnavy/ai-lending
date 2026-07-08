
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

# Check the sensitivity model
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)

print("Sensitivity model type:", type(sens_model))
print("Sensitivity model attributes:", dir(sens_model))

# Check what methods are available
if hasattr(sens_model, 'predict_proba_batch'):
    print("\nHas predict_proba_batch method")
    # Let's test it with a sample
    test_df = pd.DataFrame({
        'grade': ['C', 'D', 'E'],
        'offered_rate': [0.25, 0.28, 0.30],
        'loan_amnt': [10000, 15000, 20000],
        'annual_inc': [50000, 60000, 70000],
        'funded_amnt': [10000, 15000, 20000]
    })
    probs = sens_model.predict_proba_batch(test_df)
    print("Test acceptance probs:", probs)
