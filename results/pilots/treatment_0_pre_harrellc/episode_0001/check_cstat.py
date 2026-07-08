import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Load risk model
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'rb') as f:
    risk_data = pickle.load(f)

# Preprocess
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

X_val = risk_data['preprocessor'].transform(val)
val_risk = risk_data['model'].predict_proba(X_val)[:, 1]
val['risk_score'] = val_risk

# Check: does higher risk_score correspond to higher event rate?
print("Event rate by risk_score decile:")
val['risk_decile'] = pd.qcut(val['risk_score'], 10, labels=False)
print(val.groupby('risk_decile').agg(
    event_rate=('event', 'mean'),
    mean_risk=('risk_score', 'mean'),
    count=('event', 'count'),
    median_time=('observed_time', 'median')
))

# Check C-index more carefully
# For survival data: C-index = P(risk_i > risk_j | event_i=1, event_j=0, time_i < time_j)
# Higher risk should correspond to shorter survival time (event happens sooner)
from lifelines.utils import concordance_index as ci

# Standard: higher risk → shorter time → C-index should be > 0.5 if model is good
c1 = ci(val['risk_score'], val['event'], val['observed_time'])
print(f"\nC-index (risk_score → event, time): {c1:.4f}")

# If I negate risk_score
c2 = ci(-val['risk_score'], val['event'], val['observed_time'])
print(f"C-index (-risk_score → event, time): {c2:.4f}")

# What about just using risk_score as a binary classifier AUC?
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(val['event'], val['risk_score'])
print(f"AUC (risk_score → event): {auc:.4f}")

# Check: are higher risk_score loans actually more likely to default?
mask_high = val['risk_score'] > 0.30
mask_low = val['risk_score'] < 0.10
print(f"\nHigh risk (>{0.30:.2f}) event rate: {val.loc[mask_high, 'event'].mean():.4f}")
print(f"Low risk (<{0.10:.2f}) event rate: {val.loc[mask_low, 'event'].mean():.4f}")
print(f"Median time high risk: {val.loc[mask_high, 'observed_time'].median():.1f}")
print(f"Median time low risk: {val.loc[mask_low, 'observed_time'].median():.1f}")