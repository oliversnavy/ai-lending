import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}

# Test acceptance rates by grade and rate
print("--- Testing acceptance rates by grade and rate ---")
for grade in ['C', 'D', 'E', 'F']:
    sample = val[val['grade'] == grade].head(5000)
    for rate in [0.21, 0.25, 0.30, 0.36]:
        market_rate = market_rates[grade]
        rate_spread = np.maximum(0, rate - market_rate)
        burden = np.clip(sample['loan_amnt'].values / sample['annual_inc'].values, 0, 10)
        match = np.minimum(sample['loan_amnt'].values / np.maximum(sample['funded_amnt'].values, 1.0), 1.0)
        log_odds = 0.20 - 12.0 * rate_spread - 1.5 * burden + 0.5 * match
        probs = 1.0 / (1.0 + np.exp(-log_odds))
        print(f"  Grade {grade} (market={market_rate:.0%}) @ {rate:.0%}: avg_p={probs.mean():.4f}, median={np.median(probs):.4f}, p>0.01={np.mean(probs>0.01):.1%}")

# Now let's look at risk distribution by grade
print("\n--- Risk scores by grade ---")
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

import pickle
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'rb') as f:
    risk_data = pickle.load(f)
X_val = risk_data['preprocessor'].transform(val)
val_risk = risk_data['model'].predict_proba(X_val)[:, 1]
val['risk_score'] = val_risk

print(val.groupby('grade')['risk_score'].agg(['mean', 'std', 'count']))
print("\nEvent rate by grade:")
print(val.groupby('grade')['event'].mean())