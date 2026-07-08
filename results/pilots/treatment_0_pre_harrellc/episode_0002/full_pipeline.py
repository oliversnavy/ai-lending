import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load model
with open('risk_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']
cols = model_data['cols']

# Load sensitivity model
from data_pipeline.sensitivity_model import SensitivityModel
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

# Load val data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Encode
grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

def encode_features(df):
    df = df.copy()
    df['grade_num'] = df['grade'].map(grade_map)
    df['term_36'] = (df['term'] == '36 months').astype(int)
    home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
    df['home_own'] = df['home_ownership'].map(home_map).fillna(0).astype(int)
    verif_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
    df['verif'] = df['verification_status'].map(verif_map).fillna(0).astype(int)
    purpose_numeric = pd.Categorical(df['purpose']).codes
    df['purpose_enc'] = purpose_numeric
    df['sub_grade_letter'] = df['sub_grade'].str[0]
    sg_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['sub_grade_num'] = df['sub_grade_letter'].map(sg_map).fillna(3).astype(int)
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, 
               '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
               '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': 0}
    df['emp_len'] = df['emp_length'].map(emp_map).fillna(0).astype(int)
    return df

val_enc = encode_features(val)

# Score risk
X_val = val_enc[cols].fillna(0).values
X_val_scaled = scaler.transform(X_val)
risk_prob = model.predict_proba(X_val_scaled)[:, 1]

# Build offer dataframe
offer_df = val_enc[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
offer_df['risk_score'] = risk_prob

# Now design pricing strategy
# Key insight: We need to price above 21% (our floor) and maximize P&L
# P&L per loan = p_accept * (loan_amnt * rate * obs_time/12 - loan_amnt * event)
# We want to maximize expected P&L per dollar deployed

# Strategy: For each applicant, determine optimal rate based on grade and risk
# Higher risk = higher rate (but within 21-36% cap)
# Lower risk = lower rate (closer to 21%)

# Let's first understand acceptance by grade+rate combinations
print("Understanding acceptance patterns...")
print(f"Total val applicants: {len(val_enc)}")
print(f"Grade distribution:")
print(val_enc['grade'].value_counts())

# Create a grid of rates to test for each grade
rates_by_grade = {
    'A': [0.21, 0.24, 0.28, 0.32, 0.36],
    'B': [0.21, 0.24, 0.28, 0.32, 0.36],
    'C': [0.21, 0.24, 0.28, 0.32, 0.36],
    'D': [0.21, 0.24, 0.28, 0.32, 0.36],
    'E': [0.21, 0.24, 0.28, 0.32, 0.36],
    'F': [0.21, 0.24, 0.28, 0.32, 0.36],
    'G': [0.21, 0.24, 0.28, 0.32, 0.36],
}

# Let's try a risk-based pricing: higher risk = higher rate
# Use risk score to determine rate
def risk_based_rate(risk_score, grade):
    """Higher risk score -> higher rate (up to 36%)"""
    # Base rate depends on grade market rate
    market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
    base = market_rates.get(grade, 0.24)
    
    # Map risk score to rate between 21% and 36%
    # Higher risk -> higher rate
    rate = 0.21 + (risk_score / 1.0) * (0.36 - 0.21)
    return np.clip(rate, 0.21, 0.36)

# Test this approach
offer_df['offered_rate'] = offer_df.apply(lambda r: risk_based_rate(r['risk_score'], r['grade']), axis=1)

print("\nOffer rate stats:")
print(f"  Min: {offer_df['offered_rate'].min():.4f}")
print(f"  Max: {offer_df['offered_rate'].max():.4f}")
print(f"  Mean: {offer_df['offered_rate'].mean():.4f}")
print(f"  Median: {offer_df['offered_rate'].median():.4f}")

# Get acceptance probabilities
accept_probs = sens_model.predict_proba_batch(offer_df)
offer_df['p_accept'] = accept_probs

print(f"\nAcceptance rate: {accept_probs.mean():.4f}")
print(f"Mean offered rate: {offer_df['offered_rate'].mean():.4f}")

# Compute P&L
# We need observed_time from val data
val_enc['observed_time'] = val['observed_time'].values
val_enc['event'] = val['event'].values

offer_df['observed_time'] = val_enc['observed_time'].values
offer_df['event'] = val_enc['event'].values

# P&L calculation
offer_df['expected_principal'] = offer_df['p_accept'] * offer_df['loan_amnt']
offer_df['expected_interest'] = offer_df['p_accept'] * offer_df['loan_amnt'] * offer_df['offered_rate'] * (offer_df['observed_time'] / 12)
offer_df['expected_loss'] = offer_df['p_accept'] * offer_df['loan_amnt'] * offer_df['event']
offer_df['expected_pnl'] = offer_df['expected_interest'] - offer_df['expected_loss']

total_principal = offer_df['expected_principal'].sum()
total_pnl = offer_df['expected_pnl'].sum()
loans_funded = offer_df['p_accept'].sum()
acceptance_rate = offer_df['p_accept'].mean()

print(f"\n=== Results ===")
print(f"Total expected principal: ${total_principal:,.0f}")
print(f"Total expected P&L: ${total_pnl:,.0f}")
print(f"Expected loans funded: {loans_funded:.0f}")
print(f"Acceptance rate: {acceptance_rate:.4f}")

# Check constraints
if total_principal > 15_000_000:
    print("\nWARNING: Principal exceeds $15M cap!")
    # Greedy selection by P&L/principal ratio
    offer_df['pnl_ratio'] = offer_df['expected_pnl'] / offer_df['expected_principal'].replace(0, np.nan)
    offer_df_sorted = offer_df.sort_values('pnl_ratio', ascending=False)
    cum_principal = 0
    selected = []
    for _, row in offer_df_sorted.iterrows():
        if cum_principal + row['expected_principal'] <= 15_000_000:
            selected.append(row)
            cum_principal += row['expected_principal']
        else:
            break
    selected_df = pd.DataFrame(selected)
    total_principal = selected_df['expected_principal'].sum()
    total_pnl = selected_df['expected_pnl'].sum()
    loans_funded = selected_df['p_accept'].sum()
    print(f"Selected {len(selected_df)} loans (from {len(offer_df)}), principal: ${total_principal:,.0f}")
