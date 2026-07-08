import pandas as pd
import numpy as np

# Try to load the model with proper path setup
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

# Check if there's a module defining the sensitivity model
import importlib.util
spec = importlib.util.find_spec('sensitivity_model')
print("sensitivity_model spec:", spec)

# Try loading with sys.path
import pickle
import importlib
import sys

# First, let's see what's in the data/processed directory
import os
print("\nFiles in data/processed:")
for f in os.listdir('/home/oliversnavy/repos/ai-lending/data/processed/'):
    print(f"  {f}")