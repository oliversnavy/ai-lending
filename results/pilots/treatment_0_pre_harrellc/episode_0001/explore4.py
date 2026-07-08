import pickle
import sys
import io

# Create a dummy class to load the pickle
class SensitivityModel:
    pass

# Monkey-patch
sys.modules['__main__'].SensitivityModel = SensitivityModel

# Now try loading
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model type:", type(model))
print("Model attributes:", [a for a in dir(model) if not a.startswith('_')])
if hasattr(model, 'feature_names'):
    print("Features:", model.feature_names)
if hasattr(model, 'classes_'):
    print("Classes:", model.classes_)