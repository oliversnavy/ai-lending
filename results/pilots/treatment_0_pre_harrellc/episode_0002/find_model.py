# Let me find the SensitivityModel class definition
import sys
import os
import pickle

# Check if we can import from the project root
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

# Try to find the class
for path in sys.path:
    if path and os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.py'):
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, 'r') as fh:
                            content = fh.read()
                            if 'SensitivityModel' in content:
                                print(f"Found in: {fpath}")
                    except:
                        pass
