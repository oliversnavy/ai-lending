
# Let's find the SensitivityModel class
import os
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if f.endswith('.py') and 'sens' in f.lower():
            print(os.path.join(root, f))
