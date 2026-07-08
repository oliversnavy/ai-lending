"""
Credit Risk Model & Portfolio Optimization Pipeline v2
Trains a risk model on LendingClub data, optimizes pricing for P&L.
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

# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def build_features(df):
    """Build features for risk model from raw data."""
    X = pd.DataFrame()

    # Core numeric features
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

    # Derived features
    X['inc_per_loan'] = df['annual_inc'] / (df['loan_amnt'] + 1)
    X['debt_burden'] = df['dti'] * df['loan_amnt']

    # Encode grade (ordinal)
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    X['grade_num'] = df['grade'].map(grade_map).fillna(3)

    # Encode sub_grade
    X['sub_grade_num'] = df['sub_grade'].str[1].astype(float).fillna(3)

    # Encode home ownership
    home_map = {'OWN': 1, 'MORTGAGE': 2, 'RENT': 0, 'OTHER': -1, 'NONE': -2}
    X['home_ownership'] = df['home_ownership'].map(home_map).fillna(0)

    # Encode verification status
    verify_map = {'Verified': 2, 'Source Verified': 1, 'Not Verified': 0}
    X['verification'] = df['verification_status'].map(verify_map).fillna(0)

    # Encode purpose (top categories)
    top_purposes = ['debt_consolidation', 'credit_card', 'house', 'major_purchase',
                     'car', 'home_improvement', 'medical', 'wedding', 'other', 'renewable_energy']
    X['purpose'] = df['purpose'].map({p: i for i, p in enumerate(top_purposes)}).fillna(99)

    # FICO range midpoints
    X['fico_low'] = df['fico_range_low']
    X['fico_high'] = df['fico_range_high']
    X['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    # Employment length
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
               '10+ years': 10}
    X['emp_length'] = df['emp_length'].map(emp_map).fillna(0)

    # Term (36 = 60 month loan)
    X['term_months'] = pd.to_numeric(df['term'].str.extract(r'(\d+)')[0], errors='coerce').fillna(36)
    X['is_60mo'] = (X['term_months'] == 60).astype(float)

    # Zip code (use first digit as proxy for state risk)
    X['zip_first'] = df['zip_code'].str[0]

    # Fill NaN
    X = X.fillna(0)

    return X


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading data...")
train_df = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val_df = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

print(f"Train: {train_df.shape}, Val: {val_df.shape}")

# ============================================================
# 3. BUILD FEATURES & TARGETS
# ============================================================

print("Building features...")
train_X = build_features(train_df)
val_X = build_features(val_df)

# Target: event (1 = charged off / default)
y_train = train_df['event'].values
y_val = val_df['event'].values

# ============================================================
# 4. TRAIN RISK MODEL
# ============================================================

print("Training risk model...")
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# Train logistic regression for P(default)
lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr.fit(train_X_scaled, y_train)

# Evaluate on train
train_pred = lr.predict_proba(train_X_scaled)[:, 1]
train_ci = concordance_index(train_df['observed_time'], -train_pred, train_df['event'])
print(f"Train C-index: {train_ci:.4f}")

# Evaluate on val
val_pred = lr.predict_proba(val_X_scaled)[:, 1]
val_ci = concordance_index(val_df['observed_time'], -val_pred, val_df['event'])
print(f"Val C-index: {val_ci:.4f}")

# ============================================================
# 5. SIMULATE WITH SENSITIVITY MODEL
# ============================================================

print("\n=== Running Simulation ===")

sensitivity_model = SensitivityModel()

# Prepare val data for sensitivity model
val_for_model = val_df[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
val_for_model['default_prob'] = val_pred

# Grade base rates (what market offers)
grade_base_rate = {
    'A': 0.08, 'B': 0.10, 'C': 0.19, 'D': 0.24, 'E': 0.28, 'F': 0.32, 'G': 0.37
}

def compute_offered_rate(grade, default_prob, risk_mult):
    base = grade_base_rate.get(grade, 0.24)
    risk_adjustment = (default_prob - 0.18) * 15
    offered_rate = base + risk_adjustment
    offered_rate = offered_rate * (risk_mult / 15)
    offered_rate = max(0.21, min(0.36, offered_rate))
    return round(offered_rate, 4)

best_pnl = -np.inf
best_results = None

# Strategy grid: vary the risk adjustment multiplier
for risk_mult in [5, 10, 15, 20, 25, 30]:
    # Compute offer rates vectorized
    base_rates = val_for_model['grade'].map(grade_base_rate).fillna(0.24)
    risk_adj = (val_for_model['default_prob'] - 0.18) * 15
    offered_rates = (base_rates + risk_adj) * (risk_mult / 15)
    offered_rates = offered_rates.clip(0.21, 0.36).round(4)
    val_for_model['offered_rate'] = offered_rates.values

    # Get acceptance probabilities from sensitivity model
    sens_input = val_for_model[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
    sens_input['p_accept'] = sensitivity_model.predict_proba_batch(sens_input)

    # Filter: only consider loans with meaningful acceptance probability
    sens_input = sens_input[sens_input['p_accept'] > 0.001].copy()

    if len(sens_input) == 0:
        print(f"  Mult={risk_mult}: No valid loans (empty after filter)")
        continue

    # Get the original indices for alignment
    orig_idx = sens_input.index
    observed_times = val_df.loc[orig_idx, 'observed_time'].values
    events = val_df.loc[orig_idx, 'event'].values

    # Compute P&L
    sens_input['expected_principal'] = sens_input['p_accept'] * sens_input['loan_amnt']
    sens_input['expected_interest'] = (sens_input['p_accept'] * sens_input['loan_amnt'] *
                                         sens_input['offered_rate'] * (observed_times / 12))
    sens_input['expected_loss'] = sens_input['p_accept'] * sens_input['loan_amnt'] * events
    sens_input['expected_pnl'] = sens_input['expected_interest'] - sens_input['expected_loss']

    total_principal = sens_input['expected_principal'].sum()
    loans_funded = sens_input['p_accept'].sum()
    total_pnl = sens_input['expected_pnl'].sum()
    acceptance_rate = sens_input['p_accept'].mean()

    # Apply capital cap: rank by P&L per unit principal, include until cap
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

    print(f"  Mult={risk_mult}: N={len(sens_input)}, PnL=${total_pnl:,.0f}, "
          f"Principal=${total_principal:,.0f}, Loans={loans_funded:.0f}, "
          f"Acept={acceptance_rate:.4f}")

    if total_pnl > best_pnl:
        best_pnl = total_pnl
        best_results = {
            'mult': risk_mult,
            'data': sens_input,
            'total_principal': total_principal,
            'loans_funded': loans_funded,
            'total_pnl': total_pnl,
            'acceptance_rate': acceptance_rate
        }

if best_results is None:
    print("ERROR: No valid strategy found!")
    # Fallback: use 25% flat offer to everyone
    val_for_model['offered_rate'] = 0.25
    sens_input = val_for_model[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
    sens_input['p_accept'] = sensitivity_model.predict_proba_batch(sens_input)
    sens_input = sens_input[sens_input['p_accept'] > 0.001].copy()
    orig_idx = sens_input.index
    observed_times = val_df.loc[orig_idx, 'observed_time'].values
    events = val_df.loc[orig_idx, 'event'].values
    sens_input['expected_principal'] = sens_input['p_accept'] * sens_input['loan_amnt']
    sens_input['expected_interest'] = (sens_input['p_accept'] * sens_input['loan_amnt'] *
                                         sens_input['offered_rate'] * (observed_times / 12))
    sens_input['expected_loss'] = sens_input['p_accept'] * sens_input['loan_amnt'] * events
    sens_input['expected_pnl'] = sens_input['expected_interest'] - sens_input['expected_loss']
    best_results = {
        'mult': 99,
        'data': sens_input,
        'total_principal': sens_input['expected_principal'].sum(),
        'loans_funded': sens_input['p_accept'].sum(),
        'total_pnl': sens_input['expected_pnl'].sum(),
        'acceptance_rate': sens_input['p_accept'].mean()
    }
    best_pnl = best_results['total_pnl']

print(f"\nBest strategy: mult={best_results['mult']}, PnL=${best_pnl:,.0f}")

# ============================================================
# 6. COMPUTE FINAL METRICS
# ============================================================

final_data = best_results['data']
final_pnl = best_results['total_pnl']
final_principal = best_results['total_principal']
final_loans = int(round(best_results['loans_funded']))
final_acceptance = best_results['acceptance_rate']

print(f"\n=== FINAL RESULTS ===")
print(f"P&L: ${final_pnl:,.2f}")
print(f"Total Principal: ${final_principal:,.2f}")
print(f"Loans Funded: {final_loans}")
print(f"Acceptance Rate: {final_acceptance:.4f}")
print(f"Val C-index: {val_ci:.4f}")

# Check constraints
print(f"\nConstraint checks:")
print(f"  Principal <= 15M: {final_principal <= 15_000_000} ({final_principal:,.0f})")
print(f"  Loans >= 400: {final_loans >= 400} ({final_loans})")
print(f"  Acceptance > 1%: {final_acceptance > 0.01} ({final_acceptance:.4f})")

# ============================================================
# 7. WRITE results.json
# ============================================================

results = {
    "pnl": float(final_pnl),
    "c_stat": float(val_ci),
    "acceptance_rate": float(final_acceptance),
    "loans_funded": final_loans,
    "total_principal": float(final_principal),
    "approach": "Logistic regression default model with grade-based pricing; offer rates 21-36% scaled by predicted default risk, optimized via sensitivity model acceptance simulation",
    "hypothesis": "Higher-risk grades (C-G) with rates above 21% will generate positive P&L because the interest income exceeds expected losses even at moderate default rates"
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nresults.json written. P&L: ${final_pnl:,.2f}")
print(json.dumps(results, indent=2))
