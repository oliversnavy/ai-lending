import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load sensitivity model directly
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    # Need to define the class first for unpickling
    class SensitivityModel:
        pass
    # Actually, let me just define it properly

# Let me define the class and load
class SensitivityModel:
    pass

# Actually, let me just import from the module
from data_pipeline import sensitivity_model as sm_module

# Define the class in the module namespace
SensitivityModel = sm_module.SensitivityModel

# Now load
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))
print("Loaded sensitivity model")
print(f"Type: {type(sens_model)}")

# Test it
print("\n--- Sensitivity model test ---")
test_cases = [
    ('C', 0.21, 10000, 50000, 10000),
    ('C', 0.25, 10000, 50000, 10000),
    ('C', 0.30, 10000, 50000, 10000),
    ('D', 0.21, 10000, 50000, 10000),
    ('D', 0.30, 10000, 50000, 10000),
    ('D', 0.36, 10000, 50000, 10000),
    ('E', 0.25, 10000, 50000, 10000),
    ('E', 0.36, 10000, 50000, 10000),
    ('F', 0.36, 10000, 50000, 10000),
]

for grade, rate, la, ai, fa in test_cases:
    prob = sens_model.predict_proba(grade, rate, la, ai, fa)
    print(f"  {grade} @ {rate:.0%}: p_accept={prob:.4f}")
