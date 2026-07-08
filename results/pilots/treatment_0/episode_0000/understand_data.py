# Test sensitivity model using the query tool instead
# Let me just use the sensitivity_model_query tool for testing
# For now, let me build the full pipeline using the tool directly

# Let me first understand the data better
import pandas as pd
import numpy as np

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check int_rate distribution by grade - this tells us market rates
print("Market rates by grade:")
for g in ['C', 'D', 'E', 'F', 'G']:
    subset = val[val['grade'] == g]
    print(f"  Grade {g}: median={subset['int_rate'].median():.2f}%, mean={subset['int_rate'].mean():.2f}%, min={subset['int_rate'].min():.2f}%, max={subset['int_rate'].max():.2f}%")

print("\nVal data stats:")
print(f"Total rows: {len(val)}")
print(f"Grade distribution:")
print(val['grade'].value_counts())
print(f"\nTerm distribution:")
print(val['term'].value_counts())
print(f"\nLoan amount stats:")
print(val['loan_amnt'].describe())
print(f"\nFunded amount stats:")
print(val['funded_amnt'].describe())
