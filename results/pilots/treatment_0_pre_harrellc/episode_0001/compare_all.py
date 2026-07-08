import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Load risk model
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'rb') as f:
    risk_data = pickle.load(f)

# Preprocess val
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

# C-stat using lifelines
from lifelines.utils import concordance_index as ci
c_stat = ci(-val['risk_score'], val['event'], val['observed_time'])
print(f"C-stat: {c_stat:.4f}")

# ============================================================
# COMPREHENSIVE STRATEGY COMPARISON
# ============================================================

viable_grades = ['C', 'D', 'E', 'F']
val_v = val[val['grade'].isin(viable_grades)].copy()

# Strategy 1: Flat 21% for all, no risk filter
# Strategy 2: Flat 21%, risk filter at 0.50
# Strategy 3: Flat 21%, risk filter at 0.45
# Strategy 4: Risk-based pricing (0.21 + risk*0.20), no filter
# Strategy 5: Risk-based pricing + risk filter at 0.50
# Strategy 6: Only D/E/F, flat 21%
# Strategy 7: Only D/E, flat 21% (higher acceptance)

market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}

def simulate_strategy(df, rate_func, rate_col_name='rate'):
    """Simulate a strategy and return portfolio metrics."""
    df = df.copy()
    df['rate'] = rate_func(df)
    
    mr = df['grade'].map(market_rates).values
    spread = np.maximum(0, df['rate'].values - mr)
    burden = np.clip(df['loan_amnt'].values / np.maximum(df['annual_inc'].values, 1.0), 0, 10)
    match = np.minimum(df['loan_amnt'].values / np.maximum(df['funded_amnt'].values, 1.0), 1.0)
    log_odds = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
    p_accept = 1.0 / (1.0 + np.exp(-log_odds))
    df['p_accept'] = p_accept
    
    df['expected_principal'] = df['p_accept'] * df['loan_amnt']
    df['expected_interest'] = df['p_accept'] * df['loan_amnt'] * df['rate'] * (df['observed_time'] / 12.0)
    df['expected_loss'] = df['p_accept'] * df['loan_amnt'] * df['event']
    df['expected_pnl'] = df['expected_interest'] - df['expected_loss']
    df['pnl_per_principal'] = df['expected_pnl'] / df['expected_principal'].clip(lower=1)
    
    # Greedy selection
    df_sorted = df.sort_values('pnl_per_principal', ascending=False)
    cumulative_principal = 0
    selected = []
    for _, row in df_sorted.iterrows():
        if cumulative_principal + row['expected_principal'] <= 15_000_000:
            selected.append(row)
            cumulative_principal += row['expected_principal']
        else:
            break
    
    if not selected:
        return None
    
    sel = pd.DataFrame(selected)
    return {
        'loans': len(sel),
        'expected_principal': sel['expected_principal'].sum(),
        'expected_funded': sel['p_accept'].sum(),
        'pnl': sel['expected_pnl'].sum(),
        'acceptance_rate': sel['p_accept'].mean(),
        'pnl_per_dollar': sel['expected_pnl'].sum() / sel['expected_principal'].sum(),
        'avg_rate': sel['rate'].mean(),
        'avg_risk': sel['risk_score'].mean(),
    }

# Define strategies
strategies = {
    'flat_21_all': lambda df: np.full(len(df), 0.21),
    'flat_21_risk<0.50': lambda df: np.where(df['risk_score'] < 0.50, 0.21, np.nan),
    'flat_21_risk<0.45': lambda df: np.where(df['risk_score'] < 0.45, 0.21, np.nan),
    'flat_21_risk<0.40': lambda df: np.where(df['risk_score'] < 0.40, 0.21, np.nan),
    'flat_21_risk<0.35': lambda df: np.where(df['risk_score'] < 0.35, 0.21, np.nan),
    'risk_based_0.20': lambda df: (0.21 + df['risk_score'] * 0.20).clip(0.21, 0.36),
    'risk_based_0.15': lambda df: (0.21 + df['risk_score'] * 0.15).clip(0.21, 0.36),
    'flat_21_DE': lambda df: np.where(df['grade'].isin(['D', 'E']), 0.21, np.nan),
    'flat_21_DEF': lambda df: np.where(df['grade'].isin(['D', 'E', 'F']), 0.21, np.nan),
    'flat_21_risk<0.50_DE': lambda df: np.where((df['grade'].isin(['D', 'E'])) & (df['risk_score'] < 0.50), 0.21, np.nan),
    'flat_21_risk<0.45_DE': lambda df: np.where((df['grade'].isin(['D', 'E'])) & (df['risk_score'] < 0.45), 0.21, np.nan),
}

results = {}
for name, rate_func in strategies.items():
    try:
        result = simulate_strategy(val_v, rate_func)
        if result is not None:
            results[name] = result
            print(f"{name:30s}: loans={result['loans']:4d}, principal=${result['expected_principal']:,.0f}, "
                  f"funded={result['expected_funded']:.0f}, pnl=${result['pnl']:,.0f}, "
                  f"accept={result['acceptance_rate']:.3f}, pnl/$={result['pnl_per_dollar']:.3f}")
    except Exception as e:
        print(f"{name:30s}: ERROR - {e}")

# Find best strategy
best = max(results.items(), key=lambda x: x[1]['pnl'])
print(f"\nBest strategy: {best[0]} with P&L ${best[1]['pnl']:,.0f}")