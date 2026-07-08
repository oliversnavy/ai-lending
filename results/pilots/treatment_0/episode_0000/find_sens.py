# Find where SensitivityModel is defined
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

# Search for the class definition
import os
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                    if 'SensitivityModel' in content:
                        print(f"Found in: {path}")
            except:
                pass
