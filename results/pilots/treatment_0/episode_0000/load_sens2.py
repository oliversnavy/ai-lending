# Import SensitivityModel from the data_pipeline module
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import pandas as pd

# Load the model
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)

print("Model type:", type(sens_model))

# Test with different rates and grades
test_cases = [
    ('C', 0.21, 12000, 60000, 12000),
    ('C', 0.25, 12000, 60000, 12000),
    ('C', 0.30, 12000, 60000, 12000),
    ('D', 0.21, 12000, 60000, 12000),
    ('D', 0.25, 12000, 60000, 12000),
    ('D', 0.30, 12000, 60000, 12000),
    ('D', 0.35, 12000, 60000, 12000),
    ('E', 0.21, 12000, 60000, 12000),
    ('E', 0.25, 12000, 60000, 12000),
    ('E', 0.30, 12000, 60000, 12000),
    ('E', 0.35, 12000, 60000, 12000),
    ('F', 0.21, 12000, 60000, 12000),
    ('F', 0.25, 12000, 60000, 12000),
    ('F', 0.30, 12000, 60000, 12000),
    ('F', 0.35, 12000, 60000, 12000),
    ('F', 0.40, 12000, 60000, 12000),
    ('G', 0.25, 12000, 60000, 12000),
    ('G', 0.30, 12000, 60000, 12000),
    ('G', 0.35, 12000, 60000, 12000),
    ('G', 0.40, 12000, 60000, 12000),
]

print("\nAcceptance probabilities by grade and rate:")
for grade, rate, loan, inc, funded in test_cases:
    df = pd.DataFrame([{
        'grade': grade, 'offered_rate': rate,
        'loan_amnt': loan, 'annual_inc': inc, 'funded_amnt': funded
    }])
    prob = sens_model.predict_proba_batch(df)
    print(f"  Grade {grade}, rate {rate:.0%}: acceptance = {prob[0]:.4f}")
