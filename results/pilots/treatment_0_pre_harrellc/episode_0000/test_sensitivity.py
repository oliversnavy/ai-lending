
import pandas as pd
import numpy as np
import pickle

# Load sensitivity model
model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Test sensitivity model with sample applicants
test_cases = [
    # grade, offered_rate, loan_amnt, annual_inc, funded_amnt
    ('C', 0.21, 10000, 50000, 10000),
    ('C', 0.25, 10000, 50000, 10000),
    ('C', 0.30, 10000, 50000, 10000),
    ('C', 0.36, 10000, 50000, 10000),
    ('D', 0.21, 10000, 50000, 10000),
    ('D', 0.25, 10000, 50000, 10000),
    ('D', 0.30, 10000, 50000, 10000),
    ('D', 0.36, 10000, 50000, 10000),
    ('E', 0.21, 10000, 50000, 10000),
    ('E', 0.25, 10000, 50000, 10000),
    ('E', 0.30, 10000, 50000, 10000),
    ('E', 0.36, 10000, 50000, 10000),
    ('F', 0.21, 10000, 50000, 10000),
    ('F', 0.25, 10000, 50000, 10000),
    ('F', 0.30, 10000, 50000, 10000),
    ('F', 0.36, 10000, 50000, 10000),
]

print("Acceptance probabilities by grade and rate:")
for grade, rate, _, _, _ in test_cases:
    prob = model.predict_proba_batch(
        pd.DataFrame({'grade': [grade], 'offered_rate': [rate], 
                       'loan_amnt': [10000], 'annual_inc': [50000], 'funded_amnt': [10000]})
    )[0]
    print(f"  Grade {grade}, Rate {rate:.0%}: {prob:.4f}")
