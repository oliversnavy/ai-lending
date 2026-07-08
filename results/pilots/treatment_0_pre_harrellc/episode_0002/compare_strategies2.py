import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load everything
with open('risk_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']
cols = model_data['cols']

from data_pipeline.sensitivity_model import SensitivityModel
sens_model = pickle.load(open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb'))

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

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
X_val = val_enc[cols].fillna(0).values
X_val_scaled = scaler.transform(X_val)
risk_prob = model.predict_proba(X_val_scaled)[:, 1]

offer_df = val_enc[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
offer_df['risk_score'] = risk_prob
offer_df['observed_time'] = val['observed_time'].values
offer_df['event'] = val['event'].values

# Strategy 5: More aggressive risk-based pricing
# Use risk score more aggressively - higher risk gets much higher rates
def strategy5_rate(risk_score, grade):
    market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
    base = market_rates.get(grade, 0.24)
    # Higher risk -> much higher rate
    rate = base + risk_score * 0.20
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s5'] = offer_df.apply(lambda r: strategy5_rate(r['risk_score'], r['grade']), axis=1)

# Strategy 6: Only target grades C-G, higher rates for riskier borrowers
def strategy6_rate(risk_score, grade):
    if grade in ['A', 'B']:
        return 0.21  # Minimum rate for A/B
    # For C-G, use risk-based pricing
    base = {'C': 0.22, 'D': 0.25, 'E': 0.28, 'F': 0.32, 'G': 0.36}[grade]
    rate = base + risk_score * 0.12
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s6'] = offer_df.apply(lambda r: strategy6_rate(r['risk_score'], r['grade']), axis=1)

# Strategy 7: Aggressive - only offer to high-risk, high-return applicants
def strategy7_rate(risk_score, grade):
    if risk_score < 0.3:
        return 0.21  # Skip low-risk
    base = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}.get(grade, 0.24)
    rate = base + risk_score * 0.25
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s7'] = offer_df.apply(lambda r: strategy7_rate(r['risk_score'], r['grade']), axis=1)

# Strategy 8: Very aggressive - target only E, F, G with high rates
def strategy8_rate(risk_score, grade):
    if grade in ['A', 'B', 'C']:
        return 0.21  # Skip
    base = {'D': 0.26, 'E': 0.30, 'F': 0.34, 'G': 0.36}[grade]
    rate = base + risk_score * 0.10
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s8'] = offer_df.apply(lambda r: strategy8_rate(r['risk_score'], r['grade']), axis=1)

# Evaluate all strategies
for sname, rcol in [('S5_risk_aggressive', 'rate_s5'), ('S6_grade_filtered', 'rate_s6'),
                     ('S7_risk_threshold', 'rate_s7'), ('S8_high_grade_only', 'rate_s8')]:
    test_df = offer_df.copy()
    test_df['offered_rate'] = test_df[rcol]
    test_df['p_accept'] = sens_model.predict_proba_batch(test_df[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']])
    
    test_df['expected_principal'] = test_df['p_accept'] * test_df['loan_amnt']
    test_df['expected_interest'] = test_df['p_accept'] * test_df['loan_amnt'] * test_df['offered_rate'] * (test_df['observed_time'] / 12)
    test_df['expected_loss'] = test_df['p_accept'] * test_df['loan_amnt'] * test_df['event']
    test_df['expected_pnl'] = test_df['expected_interest'] - test_df['expected_loss']
    test_df['pnl_per_principal'] = test_df['expected_pnl'] / test_df['expected_principal'].replace(0, np.nan)
    
    test_df_sorted = test_df.sort_values('pnl_per_principal', ascending=False)
    cum_principal = 0
    selected = []
    for _, row in test_df_sorted.iterrows():
        if cum_principal + row['expected_principal'] <= 15_000_000:
            selected.append(row)
            cum_principal += row['expected_principal']
        else:
            break
    
    selected_df = pd.DataFrame(selected)
    total_pnl = selected_df['expected_pnl'].sum()
    total_principal = selected_df['expected_principal'].sum()
    loans_funded = selected_df['p_accept'].sum()
    acceptance_rate = selected_df['p_accept'].mean()
    
    print(f"{sname}: P&L=${total_pnl:,.0f}, Principal=${total_principal:,.0f}, "
          f"Loans={loans_funded:.0f}, Accept={acceptance_rate:.4f}, "
          f"Rate=[{selected_df['offered_rate'].min():.2%}, {selected_df['offered_rate'].max():.2%}]")
