import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# COMPLETE SIMULATION PIPELINE
# ============================================================

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

# ============================================================
# PRICING AND SIMULATION
# ============================================================

# Key parameters
CAPITAL_CAP = 15_000_000
MIN_VOLUME = 400
MIN_RATE = 0.21
MAX_RATE = 0.36
COST_OF_CAPITAL = 0.16
SERVICING_COST = 0.03

# Strategy: Offer at 21% floor for all viable applicants
# Filter: Grade C-F only (A/B have near-zero acceptance at 21%+)
# Additional filter: risk_score < 0.55 (avoid worst defaults)

market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}

# Filter to viable grades
viable_grades = ['C', 'D', 'E', 'F']
val_viable = val[val['grade'].isin(viable_grades)].copy()
print(f"Viable applicants: {len(val_viable)}")

# Filter by risk score
val_viable = val_viable[val_viable['risk_score'] < 0.55].copy()
print(f"After risk filter: {len(val_viable)}")

# Offer rate: 21% for all (floor rate maximizes acceptance)
val_viable['offered_rate'] = MIN_RATE

# Compute acceptance probabilities
market_rate = val_viable['grade'].map(market_rates).values
rate_spread = np.maximum(0, val_viable['offered_rate'].values - market_rate)
burden = np.clip(val_viable['loan_amnt'].values / np.maximum(val_viable['annual_inc'].values, 1.0), 0, 10)
match = np.minimum(val_viable['loan_amnt'].values / np.maximum(val_viable['funded_amnt'].values, 1.0), 1.0)

# Deterministic acceptance (no noise for expected value)
log_odds = 0.20 - 12.0 * rate_spread - 1.5 * burden + 0.5 * match
p_accept = 1.0 / (1.0 + np.exp(-log_odds))

val_viable['p_accept'] = p_accept

print(f"\nAcceptance stats: mean={p_accept.mean():.4f}, median={np.median(p_accept):.4f}")
print(f"Acceptance rate by grade:")
print(val_viable.groupby('grade')['p_accept'].mean())

# ============================================================
# P&L CALCULATION
# ============================================================

# For each loan:
# expected_principal = p_accept * loan_amnt
# expected_interest = p_accept * loan_amnt * offered_rate * (observed_time / 12)
# expected_loss = p_accept * loan_amnt * event
# expected_pnl = expected_interest - expected_loss

# But wait - observed_time is the actual observed time (censored or not).
# For expected value, we should use observed_time/12 for interest calculation.
# However, if event=1 (charged off), interest is only earned up to charge-off time.
# If event=0 (censored), interest continues until observed_time.

# Let's use observed_time for interest calculation
val_viable['expected_principal'] = val_viable['p_accept'] * val_viable['loan_amnt']
val_viable['expected_interest'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['offered_rate'] * (val_viable['observed_time'] / 12.0)
val_viable['expected_loss'] = val_viable['p_accept'] * val_viable['loan_amnt'] * val_viable['event']
val_viable['expected_pnl'] = val_viable['expected_interest'] - val_viable['expected_loss']

# Check capital usage
total_principal = val_viable['expected_principal'].sum()
print(f"\nTotal expected principal: ${total_principal:,.0f}")
print(f"Expected loans funded: {val_viable['p_accept'].sum():.0f}")
print(f"Total expected P&L: ${val_viable['expected_pnl'].sum():,.0f}")

# Apply capital cap: rank by expected_pnl / expected_principal and greedily select
val_viable['pnl_per_principal'] = val_viable['expected_pnl'] / val_viable['expected_principal'].clip(lower=1)

# Sort by P&L per dollar and select greedily
val_sorted = val_viable.sort_values('pnl_per_principal', ascending=False)

# Greedy selection
cumulative_principal = 0
selected = []
for _, row in val_sorted.iterrows():
    if cumulative_principal + row['expected_principal'] <= CAPITAL_CAP:
        selected.append(row)
        cumulative_principal += row['expected_principal']
    else:
        break

selected_df = pd.DataFrame(selected)

print(f"\n=== SELECTED PORTFOLIO ===")
print(f"Loans selected: {len(selected_df)}")
print(f"Total expected principal: ${selected_df['expected_principal'].sum():,.0f}")
print(f"Expected loans funded: {selected_df['p_accept'].sum():.0f}")
print(f"Total expected P&L: ${selected_df['expected_pnl'].sum():,.0f}")
print(f"Acceptance rate: {selected_df['p_accept'].mean():.4f}")
print(f"P&L per dollar deployed: {selected_df['expected_pnl'].sum() / selected_df['expected_principal'].sum():.4f}")

# Grade breakdown
print(f"\nPortfolio grade breakdown:")
print(selected_df.groupby('grade').agg(
    loans=('p_accept', 'count'),
    expected_principal=('expected_principal', 'sum'),
    expected_pnl=('expected_pnl', 'sum'),
    acceptance=('p_accept', 'mean')
))