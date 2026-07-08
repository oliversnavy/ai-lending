import pandas as pd
import numpy as np

# Check term column
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("term value counts:")
print(train['term'].value_counts())
print("\nhome_ownership value counts:")
print(train['home_ownership'].value_counts())
print("\nverification_status value counts:")
print(train['verification_status'].value_counts())
print("\nemp_length value counts:")
print(train['emp_length'].value_counts())
print("\npurpose value counts:")
print(train['purpose'].value_counts())