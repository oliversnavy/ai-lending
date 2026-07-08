# Let's check the sensitivity model class definition
import pickle
import inspect

# Read the source to find the class
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    data = f.read()

# Try to find the module/class info
print("First 500 bytes of pickle:", repr(data[:500]))

# Let's look for the class in the codebase
import os
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if f.endswith('.py') and 'sensitivity' in f.lower():
            print(f"Found: {os.path.join(root, f)}")
