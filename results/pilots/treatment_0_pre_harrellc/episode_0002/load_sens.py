import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel
import pickle

# Load the model
model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))
print("Model type:", type(model))
print("Model attributes:", dir(model))

# Test it
print("\nTest queries:")
print(model.query('C', 0.24, 10000, 50000, 10000))
print(model.query('D', 0.28, 10000, 50000, 10000))
print(model.query('E', 0.30, 10000, 50000, 10000))
print(model.query('F', 0.36, 10000, 50000, 10000))
print(model.query('A', 0.21, 10000, 50000, 10000))
print(model.query('B', 0.21, 10000, 50000, 10000))
