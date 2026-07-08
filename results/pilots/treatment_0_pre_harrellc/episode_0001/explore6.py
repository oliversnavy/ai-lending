import pickle
import sys
import pandas as pd
import numpy as np

class SensitivityModel:
    pass
sys.modules['__main__'].SensitivityModel = SensitivityModel

with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Check all callable attributes
print("Callable methods:")
for attr in dir(model):
    if not attr.startswith('_') and callable(getattr(model, attr)):
        print(f"  {attr}")

# Check the market_rates
print("\nmarket_rates:", model.market_rates)