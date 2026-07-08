import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending/data/processed')
import pickle

# Need to find the class definition
# Let's try loading with a custom unpickler
import io

class SensitivityModel:
    def predict_proba(self, grade, offered_rate, loan_amnt, annual_inc, funded_amnt):
        pass
    def predict_proba_batch(self, df):
        pass

# Try loading
import pickle
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    model = pickle.load(f, fix_imports=True)

print(type(model))
print(type(model).__module__)

# Check if it has the right methods
if hasattr(model, 'predict_proba_batch'):
    print("Has predict_proba_batch")
else:
    print("No predict_proba_batch")