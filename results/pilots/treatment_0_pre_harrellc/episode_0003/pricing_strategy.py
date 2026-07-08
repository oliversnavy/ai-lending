import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

from data_pipeline.sensitivity_model import SensitivityModel
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Create derived features
for df in [train, val]:
    df['term_num'] = (df['term'] == '60 months').astype(int)
    df['grade_num'] = df['grade'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

feature_cols = [
    'loan_amnt', 'funded_amnt', 'annual_inc', 'dti', 'revol_bal', 'revol_util',
    'open_acc', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'collections_12_mths_ex_med',
    'acc_now_delinq', 'total_acc', 'num_tl_120dpd_2m', 'num_tl_30dpd',
    'fico_range_low', 'fico_range_high', 'term_num', 'grade_num'
]

# Load risk model
risk_model = pickle.load(open('risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

X_val = val[feature_cols].copy()
for col in feature_cols:
    med = train[col].median()
    X_val[col] = X_val[col].fillna(med)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

val['default_prob'] = risk_model.predict_proba(scaler.transform(X_val))[:, 1]

# Compute C-index for risk model
# Use observed_time and event from val
ci = concordance_index(val['observed_time'], -val['default_prob'], val['event'])
print(f"Concordance Index: {ci:.4f}")

# Load sensitivity model
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Now design pricing strategy
# Strategy: For each applicant, find the rate that maximizes expected P&L
# Expected P&L = p_accept * (loan_amnt * rate * obs_time/12 - loan_amnt * default_prob)
# We need rate in [0.21, 0.36]

# First, let's understand the trade-off: higher rate = more interest but lower acceptance
# Let's create a grid search for each applicant

# Key: observed_time in months, default_prob from risk model
val['obs_months'] = val['observed_time']

# For each applicant, compute expected P&L at different rates
# We'll use a grid search approach

print("Building pricing strategy...")

# Let's first check the distribution of observed_time
print(f"\nObserved time stats:")
print(val['observed_time'].describe())

# Check term distribution
print(f"\nTerm distribution:")
print(val['term'].value_counts())

# For a 36-month loan, expected_interest = loan_amnt * rate * 36/12 = loan_amnt * rate * 3
# For a 60-month loan, expected_interest = loan_amnt * rate * 60/12 = loan_amnt * rate * 5
# But the loan is only held until default/observation, so actual interest depends on observed_time

# Let's compute expected P&L per loan:
# E[P&L] = p_accept * [loan_amnt * rate * (obs_time/12) - loan_amnt * default_prob]
# E[P&L] = p_accept * loan_amnt * [rate * (obs_time/12) - default_prob]

# For optimization, we need to find rate that maximizes this
# But p_accept depends on rate too

# Let's do a grid search for each applicant
# First, let's check if val has enough data to process efficiently
print(f"\nVal size: {len(val)}")

# Let's sample a subset for the grid search to understand the trade-off
sample = val.sample(10000, random_state=42)

# Compute expected P&L at different rates for sample
rates = np.arange(0.21, 0.37, 0.01)  # 21% to 36%

results_grid = []
for i, row in sample.iterrows():
    best_rate = None
    best_pnl = -np.inf
    best_pnl_info = None
    
    for rate in rates:
        p_accept = sens_model.predict_proba(
            row['grade'], rate, row['loan_amnt'], row['annual_inc'], row['funded_amnt']
        )
        if p_accept < 0.001:
            continue
        
        expected_pnl = p_accept * row['loan_amnt'] * (rate * (row['observed_time'] / 12.0) - row['default_prob'])
        
        if expected_pnl > best_pnl:
            best_pnl = expected_pnl
            best_rate = rate
            best_pnl_info = {'p_accept': p_accept, 'rate': rate, 'expected_pnl': expected_pnl}
    
    if best_pnl_info:
        results_grid.append({
            'loan_amnt': row['loan_amnt'],
            'grade': row['grade'],
            'default_prob': row['default_prob'],
            'observed_time': row['observed_time'],
            'best_rate': best_rate,
            'best_pnl': best_pnl,
            'best_p_accept': best_pnl_info['p_accept'],
        })

grid_df = pd.DataFrame(results_grid)
print(f"\nGrid search results ({len(grid_df)} applicants with valid offers):")
print(f"Best rate distribution:")
print(grid_df['best_rate'].describe())
print(f"\nAverage best rate: {grid_df['best_rate'].mean():.4f}")
print(f"Median best rate: {grid_df['best_rate'].median():.4f}")
print(f"\nAverage expected P&L per loan: {grid_df['best_pnl'].mean():.2f}")
print(f"Average p_accept: {grid_df['best_p_accept'].mean():.4f}")

# Check by grade
print("\nBest rate by grade:")
for grade in ['C', 'D', 'E', 'F', 'G']:
    subset = grid_df[grid_df['grade'] == grade]
    if len(subset) > 0:
        print(f"  {grade}: mean_rate={subset['best_rate'].mean():.4f}, mean_pnl={subset['best_pnl'].mean():.2f}, n={len(subset)}")
