import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pickle
import numpy as np
import pandas as pd
from data_pipeline.sensitivity_model import SensitivityModel

# Load the sensitivity model
sens_model = SensitivityModel()

# Test it
df_test = pd.DataFrame({
    'grade': ['C', 'D', 'E', 'F', 'A', 'B'],
    'offered_rate': [0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    'loan_amnt': [10000, 10000, 10000, 10000, 10000, 10000],
    'annual_inc': [50000, 50000, 50000, 50000, 50000, 50000],
    'funded_amnt': [8000, 8000, 8000, 8000, 8000, 8000]
})
probs = sens_model.predict_proba_batch(df_test)
print("Acceptance probs by grade at 25% offer rate:")
for g, p in zip(df_test['grade'], probs):
    print(f"  {g}: {p:.3f}")

# Also test with market rate
df_test2 = pd.DataFrame({
    'grade': ['C', 'D', 'E', 'F', 'A', 'B'],
    'offered_rate': [0.19, 0.24, 0.295, 0.34, 0.095, 0.14],
    'loan_amnt': [10000, 10000, 10000, 10000, 10000, 10000],
    'annual_inc': [50000, 50000, 50000, 50000, 50000, 50000],
    'funded_amnt': [8000, 8000, 8000, 8000, 8000, 8000]
})
probs2 = sens_model.predict_proba_batch(df_test2)
print("\nAcceptance probs at market rates:")
for g, p in zip(df_test2['grade'], probs2):
    print(f"  {g}: {p:.3f}")