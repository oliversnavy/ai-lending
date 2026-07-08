import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("term unique values:", train['term'].unique())
print("int_rate unique values:", train['int_rate'].head(10))
