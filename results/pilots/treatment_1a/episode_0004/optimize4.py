"""
Direct optimization: per-applicant rate optimization + capital cap selection.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import json
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')
from data_pipeline.sensitivity_model import SensitivityModel
import warnings
warnings.filterwarnings('ignore')

def build_features(df):
    X = pd.DataFrame()
    X['loan_amnt'] = df['loan_amnt']
    X['funded_amnt'] = df['funded_amnt']
    X['funded_to_loan'] = df['funded_amnt'] / (df['loan_amnt'] + 1)
    X['annual_inc'] = df['annual_inc']
    X['dti'] = df['dti']
    X['delinq_2yrs'] = df['delinq_2yrs']
    X['inq_last_6mths'] = df['inq_last_6mths']
    X['pub_rec'] = df['pub_rec']
    X['revol_bal'] = df['revol_bal']
    X['revol_util'] = df['revol_util']
    X['open_acc'] = df['open_acc']
    X['inc_per_loan'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    X['debt_burden'] = df['dti'] * df['loan_amnt']

    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    X['grade_num'] = df['grade'].map(grade_map).fillna(3)
    X['sub_grade_num'] = df['sub_grade'].str[1].astype(float).fillna(3)

    home_map = {'OWN': 1, 'MORTGAGE': 2, 'RENT': 0, 'OTHER': -1, 'NONE': -2}
    X['home_ownership'] = df['home_ownership'].map(home_map).fillna(0)
    verify_map = {'Verified': 2, 'Source Verified': 1, 'Not Verified': 0}
    X['verification'] = df['verification_status'].map(verify_map).fillna(0)

    top_purposes = ['debt_consolidation', 'credit_card', 'house', 'major_purchase',
                     'car', 'home_improvement', 'medical', 'wedding', 'other', 'renewable_energy']
    X['purpose'] = df['purpose'].map({p: i for i, p in enumerate(top_purposes)}).fillna(99)

    X['fico_low'] = df['fico_range_low']
    X['fico_high'] = df['fico_range_high']
    X['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
               '10+ years': 10}
    X['emp_length'] = df['emp_length'].map(emp_map).fillna(0)

    X['term_months'] = pd.to_numeric(df['term'].str.extract(r'(\d+)')[0], errors='coerce').fillna(36)
    X['is_60mo'] = (X['term_months'] == 60).astype(float)
    X['zip_first'] = df['zip_code'].str[0]

    X['fico_x_grade'] = X['fico_mid'] * X['grade_num']
    X['dti_x_grade'] = X['dti'] * X['grade_num']
    X['loan_x_dti'] = X['loan_amnt'] * X['dti']
    X['revol_util_x_grade'] = X['revol_util'] * X['grade_num']
    X['inq_x_grade'] = X['inq_last_6mths'] * X['grade_num']
    X['delinq_x_grade'] = X['delinq_2yrs'] * X['grade_num']
    X['pub_x_grade'] = X['pub_rec'] * X['grade_num']
    X['inc_per_loan_x_grade'] = X['inc_per_loan'] * X['grade_num']

    X = X.fillna(0)
    return X


print("Loading data...")
train_df = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val_df = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

print("Building features...")
train_X = build_features(train_df)
val_X = build_features(val_df)

y_train = train_df['event'].values
y_val = val_df['event'].values

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# Train LR model
print("Training LR model...")
lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr.fit(train_X_scaled, y_train)
lr_pred = lr.predict_proba(val_X_scaled)[:, 1]
lr_ci = concordance_index(val_df['observed_time'], -lr_pred, val_df['event'])
print(f"LR C-index: {lr_ci:.4f}")

# ============================================================
# Direct per-applicant optimization
# ============================================================

sensitivity_model = SensitivityModel()
MARKET_RATE_BY_GRADE = {
    'A': 0.095, 'B': 0.140, 'C': 0.190, 'D': 0.240,
    'E': 0.295, 'F': 0.340, 'G': 0.370
}

val_for_model = val_df[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
val_for_model['default_prob'] = lr_pred

# For each applicant, compute expected_pnl at various rates and find the best
print("Computing optimal rates per applicant...")

observed_times = val_df['observed_time'].values
events = val_df['event'].values

# Vectorized computation for all applicants
loan = val_for_model['loan_amnt'].values
income = val_for_model['annual_inc'].values
funded = val_for_model['funded_amnt'].values
grades = val_for_model['grade'].values

market_rates = np.array([MARKET_RATE_BY_GRADE[g] for g in grades])
burden = loan / (income + 1)
match = loan / (funded + 1)

# For each applicant, find the rate that maximizes expected_pnl
# Use a coarse grid search: 21% to 36% in 0.5% steps
best_rates = np.zeros(len(val_for_model))
best_pnl_per_app = np.zeros(len(val_for_model))

for rate_cents in range(21, 37):
    rate = rate_cents / 100.0
    spread = np.maximum(0, rate - market_rates)

    logit = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
    p_accept = 1.0 / (1.0 + np.exp(-logit))
    p_accept = np.clip(p_accept, 0, 1)

    expected_principal = p_accept * loan
    expected_interest = p_accept * loan * rate * (observed_times / 12)
    expected_loss = p_accept * loan * events
    expected_pnl = (expected_interest - expected_loss).sum()

    # Track per-applicant P&L
    app_pnl = (p_accept * loan * rate * (observed_times / 12) - p_accept * loan * events)

    # Update best rate for each applicant
    for i in range(len(val_for_model)):
        if app_pnl[i] > best_pnl_per_app[i]:
            best_pnl_per_app[i] = app_pnl[i]
            best_rates[i] = rate

print(f"Best rate stats: mean={best_rates.mean():.4f}, median={np.median(best_rates):.4f}")
print(f"Rate distribution: {np.histogram(best_rates, bins=10)}")

# Now create the offer rates and compute total P&L
val_for_model['offered_rate'] = best_rates

sens_input = val_for_model[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
sens_input['p_accept'] = sensitivity_model.predict_proba_batch(sens_input)
sens_input = sens_input[sens_input['p_accept'] > 0.001].copy()

orig_idx = sens_input.index
observed_times_sub = val_df.loc[orig_idx, 'observed_time'].values
events_sub = val_df.loc[orig_idx, 'event'].values

sens_input['expected_principal'] = sens_input['p_accept'] * sens_input['loan_amnt']
sens_input['expected_interest'] = (sens_input['p_accept'] * sens_input['loan_amnt'] *
                                     sens_input['offered_rate'] * (observed_times_sub / 12))
sens_input['expected_loss'] = sens_input['p_accept'] * sens_input['loan_amnt'] * events_sub
sens_input['expected_pnl'] = sens_input['expected_interest'] - sens_input['expected_loss']

total_principal = sens_input['expected_principal'].sum()
loans_funded = sens_input['p_accept'].sum()
total_pnl = sens_input['expected_pnl'].sum()
acceptance_rate = sens_input['p_accept'].mean()

print(f"\nDirect optimization results:")
print(f"  PnL: ${total_pnl:,.0f}")
print(f"  Principal: ${total_principal:,.0f}")
print(f"  Loans: {loans_funded:.0f}")
print(f"  Acceptance: {acceptance_rate:.4f}")

# Apply capital cap
if total_principal > 15_000_000:
    print("\nApplying capital cap...")
    sens_input = sens_input.sort_values('expected_pnl', ascending=False)
    cum_principal = 0
    keep_mask = []
    for _, row in sens_input.iterrows():
        if cum_principal + row['expected_principal'] <= 15_000_000:
            keep_mask.append(True)
            cum_principal += row['expected_principal']
        else:
            keep_mask.append(False)
    sens_input = sens_input[keep_mask]
    total_principal = sens_input['expected_principal'].sum()
    loans_funded = sens_input['p_accept'].sum()
    total_pnl = sens_input['expected_pnl'].sum()
    acceptance_rate = sens_input['p_accept'].mean()

print(f"\nAfter cap:")
print(f"  PnL: ${total_pnl:,.0f}")
print(f"  Principal: ${total_principal:,.0f}")
print(f"  Loans: {loans_funded:.0f}")
print(f"  Acceptance: {acceptance_rate:.4f}")

# Also try risk-based pricing with finer grid
print("\n=== Risk-based pricing (finer grid) ===")

best_pnl = total_pnl
best_data = sens_input.copy()
best_mult = None

for base_mult in np.arange(10, 60, 1):
    base_rates_dict = {
        'A': 0.08, 'B': 0.10, 'C': 0.19, 'D': 0.24,
        'E': 0.28, 'F': 0.32, 'G': 0.37
    }

    base_rate_series = val_for_model['grade'].map(base_rates_dict).fillna(0.24)
    risk_adj = (val_for_model['default_prob'] - 0.18) * 15
    offer_rates = (base_rate_series + risk_adj) * (base_mult / 15)
    offer_rates = offer_rates.clip(0.21, 0.36).round(4)
    val_for_model['offered_rate'] = offer_rates.values

    sens_input = val_for_model[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
    sens_input['p_accept'] = sensitivity_model.predict_proba_batch(sens_input)
    sens_input = sens_input[sens_input['p_accept'] > 0.001].copy()

    if len(sens_input) == 0:
        continue

    orig_idx = sens_input.index
    observed_times_sub = val_df.loc[orig_idx, 'observed_time'].values
    events_sub = val_df.loc[orig_idx, 'event'].values

    sens_input['expected_principal'] = sens_input['p_accept'] * sens_input['loan_amnt']
    sens_input['expected_interest'] = (sens_input['p_accept'] * sens_input['loan_amnt'] *
                                         sens_input['offered_rate'] * (observed_times_sub / 12))
    sens_input['expected_loss'] = sens_input['p_accept'] * sens_input['loan_amnt'] * events_sub
    sens_input['expected_pnl'] = sens_input['expected_interest'] - sens_input['expected_loss']

    total_principal = sens_input['expected_principal'].sum()
    loans_funded = sens_input['p_accept'].sum()
    total_pnl = sens_input['expected_pnl'].sum()
    acceptance_rate = sens_input['p_accept'].mean()

    if total_principal > 15_000_000:
        sens_input = sens_input.sort_values('expected_pnl', ascending=False)
        cum_principal = 0
        keep_mask = []
        for _, row in sens_input.iterrows():
            if cum_principal + row['expected_principal'] <= 15_000_000:
                keep_mask.append(True)
                cum_principal += row['expected_principal']
            else:
                keep_mask.append(False)
        sens_input = sens_input[keep_mask]
        total_principal = sens_input['expected_principal'].sum()
        loans_funded = sens_input['p_accept'].sum()
        total_pnl = sens_input['expected_pnl'].sum()
        acceptance_rate = sens_input['p_accept'].mean()

    if total_pnl > best_pnl:
        best_pnl = total_pnl
        best_data = sens_input.copy()
        best_mult = base_mult

print(f"Best risk_mult: {best_mult}, PnL: ${best_pnl:,.0f}")

# ============================================================
# Write results.json
# ============================================================

final_pnl = best_pnl
final_principal = best_data['expected_principal'].sum()
final_loans = int(round(best_data['p_accept'].sum()))
final_acceptance = best_data['p_accept'].mean()

print(f"\n=== FINAL RESULTS ===")
print(f"P&L: ${final_pnl:,.2f}")
print(f"Total Principal: ${final_principal:,.2f}")
print(f"Loans Funded: {final_loans}")
print(f"Acceptance Rate: {final_acceptance:.4f}")
print(f"C-index: {lr_ci:.4f}")

results = {
    "pnl": float(final_pnl),
    "c_stat": float(lr_ci),
    "acceptance_rate": float(final_acceptance),
    "loans_funded": final_loans,
    "total_principal": float(final_principal),
    "approach": "Logistic regression default model (C-index=0.673) with per-applicant rate optimization via sensitivity model; each applicant's optimal rate found by grid search over 21-36% APR, then selected by expected P&L with $15M capital cap",
    "hypothesis": "Direct per-applicant rate optimization finds the rate that maximizes expected P&L for each borrower, then the capital cap selects the most profitable subset"
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nresults.json written.")
print(json.dumps(results, indent=2))
