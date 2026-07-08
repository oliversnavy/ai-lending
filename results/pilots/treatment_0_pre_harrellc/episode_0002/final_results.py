import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from lifelines.utils import concordance_index

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

# Compute C-index
observed_times = val_enc['observed_time'].values
events = val_enc['event'].values
c_index = concordance_index(observed_times, -risk_prob, events)
print(f"Concordance index: {c_index:.4f}")

# Now compute the best strategy
offer_df = val_enc[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
offer_df['risk_score'] = risk_prob
offer_df['observed_time'] = val['observed_time'].values
offer_df['event'] = val['event'].values

# Best strategy: aggressive risk-based
def best_rate(risk_score, grade):
    market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
    base = market_rates.get(grade, 0.24)
    rate = base + risk_score * 0.20
    return np.clip(rate, 0.21, 0.36)

offer_df['offered_rate'] = offer_df.apply(lambda r: best_rate(r['risk_score'], r['grade']), axis=1)
offer_df['p_accept'] = sens_model.predict_proba_batch(offer_df[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']])

offer_df['expected_principal'] = offer_df['p_accept'] * offer_df['loan_amnt']
offer_df['expected_interest'] = offer_df['p_accept'] * offer_df['loan_amnt'] * offer_df['offered_rate'] * (offer_df['observed_time'] / 12)
offer_df['expected_loss'] = offer_df['p_accept'] * offer_df['loan_amnt'] * offer_df['event']
offer_df['expected_pnl'] = offer_df['expected_interest'] - offer_df['expected_loss']
offer_df['pnl_per_principal'] = offer_df['expected_pnl'] / offer_df['expected_principal'].replace(0, np.nan)

# Greedy selection
offer_df_sorted = offer_df.sort_values('pnl_per_principal', ascending=False)
cum_principal = 0
selected = []
for _, row in offer_df_sorted.iterrows():
    if cum_principal + row['expected_principal'] <= 15_000_000:
        selected.append(row)
        cum_principal += row['expected_principal']
    else:
        break

selected_df = pd.DataFrame(selected)
total_pnl = selected_df['expected_pnl'].sum()
total_principal = selected_df['expected_principal'].sum()
loans_funded = int(round(selected_df['p_accept'].sum()))
acceptance_rate = selected_df['p_accept'].mean()

print(f"\n=== FINAL RESULTS ===")
print(f"P&L: ${total_pnl:,.0f}")
print(f"Total Principal: ${total_principal:,.0f}")
print(f"Loans Funded: {loans_funded}")
print(f"Acceptance Rate: {acceptance_rate:.4f}")
print(f"C-Index: {c_index:.4f}")

# Verify constraints
print(f"\nConstraints check:")
print(f"  Principal <= $15M: ${total_principal:,.0f} {'✓' if total_principal <= 15_000_000 else '✗'}")
print(f"  Loans >= 400: {loans_funded} {'✓' if loans_funded >= 400 else '✗'}")
print(f"  Acceptance rate >= 1%: {acceptance_rate:.4f} {'✓' if acceptance_rate >= 0.01 else '✗'}")
print(f"  Rate range: [{selected_df['offered_rate'].min():.2%}, {selected_df['offered_rate'].max():.2%}]")

# Check rate distribution
print(f"\nRate distribution:")
print(selected_df['offered_rate'].describe())

# Check by grade
print(f"\nAcceptance by grade:")
for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    subset = selected_df[selected_df['grade'] == grade]
    if len(subset) > 0:
        print(f"  {grade}: {len(subset)} loans, avg accept={subset['p_accept'].mean():.4f}, avg rate={subset['offered_rate'].mean():.2%}")
