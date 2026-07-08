import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load model
with open('risk_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']
cols = model_data['cols']

# Load validation data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Load sensitivity model
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Encode validation data same way
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

def encode_features(df):
    df = df.copy()
    df['grade_num'] = df['grade'].map(grade_map)
    df['term_36'] = (df['term'] == '36 months').astype(int)
    home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    df['home_own'] = df['home_ownership'].map(home_map).fillna(0).astype(int)
    verif_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
    df['verif'] = df['verification_status'].map(verif_map).fillna(0).astype(int)
    purpose_numeric = pd.Categorical(df['purpose']).codes
    df['purpose_enc'] = purpose_numeric
    df['sub_grade_letter'] = df['sub_grade'].str[0]
    sg_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['sub_grade_num'] = df['sub_grade_letter'].map(sg_map).fillna(3).astype(int)
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, 
               '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
               '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': 0}
    df['emp_len'] = df['emp_length'].map(emp_map).fillna(0).astype(int)
    return df

val_enc = encode_features(val)

# Score risk
X_val = val_enc[cols].fillna(0).values
X_val_scaled = scaler.transform(X_val)
risk_prob = model.predict_proba(X_val_scaled)[:, 1]

print("Risk score stats:")
print(f"  Mean: {risk_prob.mean():.4f}")
print(f"  Std: {risk_prob.std():.4f}")
print(f"  Min: {risk_prob.min():.4f}")
print(f"  Max: {risk_prob.max():.4f}")

# Now let's understand the sensitivity model behavior
# Test a few scenarios
print("\nTesting sensitivity model:")
test_cases = [
    # (grade, rate, loan_amnt, annual_inc, funded_amnt)
    ('C', 0.21, 10000, 50000, 10000),
    ('C', 0.24, 10000, 50000, 10000),
    ('C', 0.28, 10000, 50000, 10000),
    ('D', 0.21, 10000, 50000, 10000),
    ('D', 0.28, 10000, 50000, 10000),
    ('E', 0.24, 10000, 50000, 10000),
    ('E', 0.30, 10000, 50000, 10000),
    ('F', 0.24, 10000, 50000, 10000),
    ('F', 0.36, 10000, 50000, 10000),
    ('A', 0.21, 10000, 50000, 10000),
    ('B', 0.21, 10000, 50000, 10000),
]

for grade, rate, loan, inc, funded in test_cases:
    prob = sens_model_query(grade, rate, loan, inc, funded)
    print(f"  {grade} @ {rate:.0%}: {prob}")
