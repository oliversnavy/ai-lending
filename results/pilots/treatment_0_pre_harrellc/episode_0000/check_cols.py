
import pandas as pd

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
print("All columns:", list(train.columns))
