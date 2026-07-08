# Define the SensitivityModel class so pickle can load it
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending/data/processed')

class SensitivityModel:
    pass

import pickle

with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(type(model))
print(dir(model))