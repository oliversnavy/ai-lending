# Let's look for the source of the sensitivity model
import os
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if 'sensitivity' in f.lower() or 'sensitivity_model' in f.lower():
            print(os.path.join(root, f))