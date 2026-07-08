import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import concordance_index

# ============================================================
# FULL PIPELINE WITH RESULTS
# ============================================================

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

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

# ============================================================
# C-STAT: Concordance index
# ============================================================
# C-index measures if the model's risk scores correctly rank-order the outcomes
# For survival data, we use (risk_score, event, observed_time)
# Higher risk_score should correspond to shorter observed_time (earlier event)
# But since event=1 means charged-off (bad), and we want higher risk → more likely to default

# The concordance_index from sklearn doesn't handle censoring well.
# Let's use a simpler approach: for pairs where one has an event and the other doesn't,
# check if the one with event has higher risk score.

# Actually, the task says "c_stat" = concordance index. Let me use the standard definition.
# For binary classification, C-index = AUC.
# For survival, C-index considers pairs where one event occurred before the other.

# Let me compute it using lifelines or manually
try:
    from lifelines.utils import concordance_index as ci
    c_stat = ci(-val['risk_score'], val['event'], val['observed_time'])
    print(f"C-stat (lifelines): {c_stat:.4f}")
except:
    # Manual C-index for binary outcomes
    # C-index = fraction of concordant pairs
    # For each pair (i,j) where event_i=1 and event_j=0:
    #   concordant if risk_score_i > risk_score_j
    mask_event = val['event'] == 1
    mask_censored = val['event'] == 0
    if mask_event.sum() > 0 and mask_censored.sum() > 0:
        c_stat_manual = np.mean(val.loc[mask_event, 'risk_score'].values > val.loc[mask_censored, 'risk_score'].values)
        print(f"C-stat (manual): {c_stat_manual:.4f}")
    else:
        c_stat_manual = np.nan
        print("C-stat: N/A")

# ============================================================
# PRICING STRATEGY - Risk-based pricing
# ============================================================
# Strategy: Offer rate based on risk_score and grade
# Higher risk → higher rate, but capped at 36%
# Lower risk → lower rate (down to 21%)

# For each applicant:
# base_rate = 0.21 (floor)
# risk_adjustment = risk_score * 0.30 (scale to 0-30%)
# grade_adjustment = market_rate[grade] - 0.21 (if market rate > 21%, add it)
# offered_rate = min(0.36, base_rate + risk_adjustment + grade_adjustment)

# Actually, let me try a simpler approach:
# offered_rate = 0.21 + risk_score * 0.35, clamped to [0.21, 0.36]
# This means:
#   risk=0 → 21% (floor)
#   risk=0.2 → 28%
#   risk=0.33 → 32.5%
#   risk=0.5 → 38.5% → capped at 36%

# But this might reduce acceptance too much for high-risk loans.
# Let me try: offered_rate = 0.21 + risk_score * 0.15, clamped to [0.21, 0.36]
# This is more conservative:
#   risk=0 → 21%
#   risk=0.2 → 24%
#   risk=0.5 → 28.5%
#   risk=1 → 36% (cap)

# Let me try both and compare

viable_grades = ['C', 'D', 'E', 'F']
val_v = val[val['grade'].isin(viable_grades)].copy()

# Strategy 1: Flat 21% for all
val_v['offered_rate'] = 0.21
val_v['strategy'] = 'flat_21'

# Strategy 2: Risk-based pricing
val_v['offered_rate_rb'] = (0.21 + val_v['risk_score'] * 0.20).clip(0.21, 0.36)
val_v['strategy'] = 'risk_based'

# Compute acceptance for both strategies
market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}

for strat in ['flat_21', 'risk_based']:
    if strat == 'flat_21':
        rate_col = 'offered_rate'
    else:
        rate_col = 'offered_rate_rb'
    
    val_v['rate'] = val_v[rate_col].values
    mr = val_v['grade'].map(market_rates).values
    spread = np.maximum(0, val_v['rate'].values - mr)
    burden = np.clip(val_v['loan_amnt'].values / np.maximum(val_v['annual_inc'].values, 1.0), 0, 10)
    match = np.minimum(val_v['loan_amnt'].values / np.maximum(val_v['funded_amnt'].values, 1.0), 1.0)
    log_odds = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
    p_accept = 1.0 / (1.0 + np.exp(-log_odds))
    val_v['p_accept'] = p_accept
    
    # P&L
    val_v['expected_principal'] = val_v['p_accept'] * val_v['loan_amnt']
    val_v['expected_interest'] = val_v['p_accept'] * val_v['loan_amnt'] * val_v['rate'] * (val_v['observed_time'] / 12.0)
    val_v['expected_loss'] = val_v['p_accept'] * val_v['loan_amnt'] * val_v['event']
    val_v['expected_pnl'] = val_v['expected_interest'] - val_v['expected_loss']
    val_v['pnl_per_principal'] = val_v['expected_pnl'] / val_v['expected_principal'].clip(lower=1)
    
    # Greedy selection
    val_sorted = val_v.sort_values('pnl_per_principal', ascending=False)
    cumulative_principal = 0
    selected = []
    for _, row in val_sorted.iterrows():
        if cumulative_principal + row['expected_principal'] <= 15_000_000:
            selected.append(row)
            cumulative_principal += row['expected_principal']
        else:
            break
    
    sel = pd.DataFrame(selected)
    print(f"\nStrategy: {strat}")
    print(f"  Selected: {len(sel)} loans")
    print(f"  Expected principal: ${sel['expected_principal'].sum():,.0f}")
    print(f"  Expected funded: {sel['p_accept'].sum():.0f}")
    print(f"  Total P&L: ${sel['expected_pnl'].sum():,.0f}")
    print(f"  Acceptance rate: {sel['p_accept'].mean():.4f}")
    print(f"  P&L per dollar: {sel['expected_pnl'].sum() / sel['expected_principal'].sum():.4f}")