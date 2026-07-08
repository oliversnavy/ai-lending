
# Try to find the SensitivityModel class
import sys
import os

# Look for the class definition
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if f.endswith('.py') and not f.startswith('.'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                    if 'SensitivityModel' in content:
                        print(f"Found in: {path}")
            except:
                pass
