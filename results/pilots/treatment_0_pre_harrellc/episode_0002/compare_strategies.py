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

# Build offer dataframe with risk scores
offer_df = val_enc[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
offer_df['risk_score'] = risk_prob
offer_df['observed_time'] = val['observed_time'].values
offer_df['event'] = val['event'].values

# STRATEGY 1: Risk-based pricing (higher risk = higher rate)
def strategy1_rate(risk_score, grade):
    market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
    base = market_rates.get(grade, 0.24)
    rate = 0.21 + risk_score * (0.36 - 0.21)
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s1'] = offer_df.apply(lambda r: strategy1_rate(r['risk_score'], r['grade']), axis=1)

# STRATEGY 2: Fixed rate for all (24%)
offer_df['rate_s2'] = 0.24

# STRATEGY 3: Grade-based pricing (higher grade = lower rate)
def strategy3_rate(grade):
    rates = {'A': 0.21, 'B': 0.22, 'C': 0.24, 'D': 0.27, 'E': 0.30, 'F': 0.33, 'G': 0.36}
    return rates.get(grade, 0.24)

offer_df['rate_s3'] = offer_df['grade'].map(strategy3_rate)

# STRATEGY 4: Risk + grade combined
def strategy4_rate(risk_score, grade):
    market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
    base = market_rates.get(grade, 0.24)
    # Higher risk -> rate closer to 36%, lower risk -> closer to 21%
    # But also consider that higher risk borrowers have higher default
    rate = base + risk_score * 0.15  # add up to 15% on top of market rate
    return np.clip(rate, 0.21, 0.36)

offer_df['rate_s4'] = offer_df.apply(lambda r: strategy4_rate(r['risk_score'], r['grade']), axis=1)

# Evaluate each strategy
for sname, rcol in [('S1_risk_based', 'rate_s1'), ('S2_fixed_24', 'rate_s2'), 
                     ('S3_grade_based', 'rate_s3'), ('S4_risk_grade', 'rate_s4')]:
    test_df = offer_df.copy()
    test_df['offered_rate'] = test_df[rcol]
    test_df['p_accept'] = sens_model.predict_proba_batch(test_df[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']])
    
    test_df['expected_principal'] = test_df['p_accept'] * test_df['loan_amnt']
    test_df['expected_interest'] = test_df['p_accept'] * test_df['loan_amnt'] * test_df['offered_rate'] * (test_df['observed_time'] / 12)
    test_df['expected_loss'] = test_df['p_accept'] * test_df['loan_amnt'] * test_df['event']
    test_df['expected_pnl'] = test_df['expected_interest'] - test_df['expected_loss']
    test_df['pnl_per_principal'] = test_df['expected_pnl'] / test_df['expected_principal'].replace(0, np.nan)
    
    # Greedy selection by pnl_per_principal
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
          f"Rate range=[{selected_df['offered_rate'].min():.2%}, {selected_df['offered_rate'].max():.2%}]")
