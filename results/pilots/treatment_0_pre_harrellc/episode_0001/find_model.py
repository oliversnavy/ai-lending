import os
import subprocess

# Search for SensitivityModel class definition
result = subprocess.run(['grep', '-r', 'class SensitivityModel', '/home/oliversnavy/repos/ai-lending/'], 
                       capture_output=True, text=True)
print("Search result:")
print(result.stdout)
print("stderr:", result.stderr)