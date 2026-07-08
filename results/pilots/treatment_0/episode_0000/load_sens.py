# Load the sensitivity model directly in code_executor
import pickle
import pandas as pd

with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)

print("Sensitivity model type:", type(sens_model))
print("Model attributes:", dir(sens_model))

# Test with a single applicant
test_df = pd.DataFrame([{
    'grade': 'C', 'offered_rate': 0.21,
    'loan_amnt': 12000, 'annual_inc': 60000, 'funded_amnt': 12000
}])

prob = sens_model.predict_proba_batch(test_df)
print(f"\nGrade C, rate 21%: acceptance = {prob[0]:.4f}")

# Test more points
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

print("\nAcceptance probabilities:")
for grade, rate, loan, inc, funded in test_cases:
    df = pd.DataFrame([{
        'grade': grade, 'offered_rate': rate,
        'loan_amnt': loan, 'annual_inc': inc, 'funded_amnt': funded
    }])
    prob = sens_model.predict_proba_batch(df)
    print(f"  Grade {grade}, rate {rate:.0%}: acceptance = {prob[0]:.4f}")
