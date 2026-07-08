# Use the query function instead to test
from sensitivity_model_query import *

# Actually let me check what the sensitivity_model_query expects
import subprocess
result = subprocess.run(['python', '-c', '''
import pickle
import sys
sys.path.insert(0, "/home/oliversnavy/repos/ai-lending/data/processed")
with open("sensitivity_model.pkl", "rb") as f:
    model = pickle.load(f)
print(type(model))
print(dir(model))
'''], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)