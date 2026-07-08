import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load everything
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Load risk model
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'rb') as f:
    risk_data = pickle.load(f)
risk_model = risk_data['model']
preprocessor = risk_data['preprocessor']

# Load sensitivity model
class SensitivityModel:
    pass
import sys
sys.modules['__main__'].SensitivityModel = SensitivityModel
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)

# Preprocess val data same way
val['term_months'] = val['term'].str.extract(r'(\d+)').astype(float)
def emp_to_years(emp_str):
    if pd.isna(emp_str): return np.nan
    emp_str = str(emp_str).strip()
    if emp_str == '< 1 year': return 0.5
    if emp_str == '10+ years': return 10.0
    try: return float(emp_str.split()[0])
    except: return np.nan
val['emp_years'] = val['emp_length'].apply(emp_to_years)
for df in [val]:
    df['loan_to_inc'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    df['funded_to_requested'] = df['funded_amnt'] / df['loan_amnt'].clip(lower=1)
    df['dti_ratio'] = df['dti'] / 100.0
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_ord'] = df['grade'].map(grade_map)

numeric_features = [
    'loan_amnt', 'funded_amnt', 'term_months', 'int_rate', 
    'fico_range_low', 'fico_range_high', 'fico_mid', 'dti', 'annual_inc', 
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 
    'open_acc', 'loan_to_inc', 'funded_to_requested', 'dti_ratio',
    'emp_years', 'grade_ord'
]
cat_features = ['sub_grade', 'emp_length', 'home_ownership', 
                'verification_status', 'purpose']

X_val = preprocessor.transform(val)
val_risk = risk_model.predict_proba(X_val)[:, 1]
val['risk_score'] = val_risk

print(f"Val risk score: mean={val_risk.mean():.4f}, std={val_risk.std():.4f}")
print(f"Val shape: {val.shape}")

# --- PRICING STRATEGY ---
# For each applicant, determine offered_rate based on risk score and grade
# Higher risk → higher rate (up to 36%)
# Must be in [0.21, 0.36]

# Cost of capital = 16%, so break-even rate ≈ 21%
# We want to price above 21% to make profit on good loans, but not so high that acceptance drops

# Strategy: risk-based pricing
# Base rate at market rate for grade, but with risk adjustment
# If risk_score > 0.25 → offer 36% (max)
# If risk_score > 0.15 → offer 28%
# If risk_score > 0.10 → offer 24%
# If risk_score > 0.05 → offer 21% (floor)
# Otherwise → don't offer (too safe, low margin)

# Better: continuous mapping from risk score to rate
# rate = 0.21 + (risk_score - 0.05) * factor, clamped to [0.21, 0.36]

# Let's try: rate = 0.21 + risk_score * 0.35, clamped to [0.21, 0.36]
# This gives: risk=0 → 21%, risk=0.5 → 38.5% → capped at 36%, risk=1 → 56% → capped at 36%

# Actually, let's be smarter. We need to consider:
# 1. Acceptance probability (from sensitivity model)
# 2. Expected P&L = p_accept * (interest - loss)
# 3. Capital constraint

# Let me first check acceptance rates for different rate levels
print("\n--- Testing acceptance rates by grade and rate ---")
market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}

for grade in ['C', 'D', 'E', 'F']:
    sample = val[val['grade'] == grade].head(1000)
    for rate in [0.21, 0.25, 0.30, 0.36]:
        test_df = pd.DataFrame({
            'grade': [grade]*1000,
            'offered_rate': [rate]*1000,
            'loan_amnt': sample['loan_amnt'].values,
            'annual_inc': sample['annual_inc'].values,
            'funded_amnt': sample['funded_amnt'].values
        })
        # Manually compute acceptance
        market_rate = market_rates[grade]
        rate_spread = np.maximum(0, rate - market_rate)
        burden = sample['loan_amnt'].values / sample['annual_inc'].values.clip(lower=1)
        match = np.minimum(sample['loan_amnt'].values / sample['funded_amnt'].values.clip(lower=1), 1.0)
        # Use deterministic version (no noise) for stable estimates
        log_odds = 0.20 - 12.0 * rate_spread - 1.5 * burden + 0.5 * match
        probs = 1.0 / (1.0 + np.exp(-log_odds))
        print(f"  Grade {grade} @ {rate:.0%}: avg p_accept = {probs.mean():.4f}, median = {np.median(probs):.4f}")