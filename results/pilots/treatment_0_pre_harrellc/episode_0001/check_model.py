import pickle
import os

# Check what files exist in the working directory
files = os.listdir('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/')
print("Files in working directory:")
for f in sorted(files):
    print(f"  {f}")

# Check the risk model
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'rb') as f:
    risk_data = pickle.load(f)

print(f"\nModel type: {type(risk_data['model']).__name__}")
print(f"Preprocessor type: {type(risk_data['preprocessor']).__name__}")
print(f"Feature names: {len(risk_data.get('feature_names', []))} features")