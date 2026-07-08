"""
Optimization pipeline: try gradient boosting + finer pricing grid.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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

    # Additional interactions
    X['fico_x_grade'] = X['fico_mid'] * X['grade_num']
    X['dti_x_grade'] = X['dti'] * X['grade_num']
    X['loan_x_dti'] = X['loan_amnt'] * X['dti']
    X['revol_util_x_grade'] = X['revol_util'] * X['grade_num']
    X['inq_x_grade'] = X['inq_last_6mths'] * X['grade_num']
    X['delinq_x_grade'] = X['delinq_2yrs'] * X['grade_num']
    X['pub_x_grade'] = X['pub_rec'] * X['grade_num']

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

# ============================================================
# Try both Logistic Regression and Gradient Boosting
# ============================================================

scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# Logistic Regression
lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
lr.fit(train_X_scaled, y_train)
lr_val_pred = lr.predict_proba(val_X_scaled)[:, 1]
lr_ci = concordance_index(val_df['observed_time'], -lr_val_pred, val_df['event'])
print(f"LR C-index: {lr_ci:.4f}")

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    min_samples_leaf=50, subsample=0.8
)
gb.fit(train_X_scaled, y_train)
gb_val_pred = gb.predict_proba(val_X_scaled)[:, 1]
gb_ci = concordance_index(val_df['observed_time'], -gb_val_pred, val_df['event'])
print(f"GB C-index: {gb_ci:.4f}")

# Also try blending
blend_pred = 0.5 * lr_val_pred + 0.5 * gb_val_pred
blend_ci = concordance_index(val_df['observed_time'], -blend_pred, val_df['event'])
print(f"Blend C-index: {blend_ci:.4f}")

# Pick best model
if gb_ci >= lr_ci and gb_ci >= blend_ci:
    print("Using GB model")
    final_pred = gb_val_pred
    final_ci = gb_ci
elif blend_ci > gb_ci:
    print("Using blend model")
    final_pred = blend_pred
    final_ci = blend_ci
else:
    print("Using LR model")
    final_pred = lr_val_pred
    final_ci = lr_ci

# ============================================================
# Sensitivity model
# ============================================================

sensitivity_model = SensitivityModel()

val_for_model = val_df[['grade', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
val_for_model['default_prob'] = final_pred

# Market rates by grade
MARKET_RATE_BY_GRADE = {
    'A': 0.095, 'B': 0.140, 'C': 0.190, 'D': 0.240,
    'E': 0.295, 'F': 0.340, 'G': 0.370
}

# ============================================================
# Grid search over pricing strategies
# ============================================================

print("\n=== Detailed Grid Search ===")

results_list = []

# Strategy 1: Risk-based pricing with different multipliers
for base_mult in [10, 15, 20, 25, 30, 40, 50]:
    # Base rate by grade
    base_rates = {
        'A': 0.08, 'B': 0.10, 'C': 0.19, 'D': 0.24,
        'E': 0.28, 'F': 0.32, 'G': 0.37
    }

    base_rate_series = val_for_model['grade'].map(base_rates).fillna(0.24)
    risk_adj = (val_for_model['default_prob'] - 0.18) * 15
    offer_rates = (base_rate_series + risk_adj) * (base_mult / 15)
    offer_rates = offer_rates.clip(0.21, 0.36).round(4)
    val_for_model['offered_rate'] = offer_rates.values

    sens_input = val_for_model[['grade', 'offered_rate', 'loan_amnt', 'annual_inc', 'funded_amnt']].copy()
    sens_input['p_accept'] = sensitivity_model.predict_proba_batch(sens_input)
    sens_input = sens_input[sens_input['p_accept'] > 0.001].copy()

    if len(sens_input) == 0:
        continue

    # Get aligned observed_time and event
    orig_idx = sens_input.index
    observed_times = val_df.loc[orig_idx, 'observed_time'].values
    events = val_df.loc[orig_idx, 'event'].values

    sens_input['expected_principal'] = sens_input['p_accept'] * sens_input['loan_amnt']
    sens_input['expected_interest'] = (sens_input['p_accept'] * sens_input['loan_amnt'] *
                                         sens_input['offered_rate'] * (observed_times / 12))
    sens_input['expected_loss'] = sens_input['p_accept'] * sens_input['loan_amnt'] * events
    sens_input['expected_pnl'] = sens_input['expected_interest'] - sens_input['expected_loss']

    total_principal = sens_input['expected_principal'].sum()
    loans_funded = sens_input['p_accept'].sum()
    total_pnl = sens_input['expected_pnl'].sum()
    acceptance_rate = sens_input['p_accept'].mean()

    # Apply capital cap
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

    results_list.append({
        'type': 'risk_mult',
        'param': base_mult,
        'pnl': total_pnl,
        'principal': total_principal,
        'loans': loans_funded,
        'acceptance_rate': acceptance_rate,
        'n': len(sens_input)
    })

# Strategy 2: Direct optimization - for each applicant, find the rate that maximizes expected_pnl
print("\n--- Direct optimization per applicant ---")

# Sample a subset for direct optimization (too expensive for full dataset)
np.random.seed(42)
sample_idx = np.random.choice(len(val_for_model), min(50000, len(val_for_model)), replace=False)
sample_data = val_for_model.iloc[sample_idx].copy()
sample_data['default_prob'] = final_pred[sample_idx]

# For each applicant, find the rate that maximizes expected_pnl
def find_optimal_rate(row, default_prob, observed_time, event):
    """Find the rate in [0.21, 0.36] that maximizes expected_pnl."""
    best_pnl = -np.inf
    best_rate = 0.21

    for rate_cents in range(21, 37):
        rate = rate_cents / 100.0

        # Estimate acceptance probability using the sensitivity model
        # We need to approximate: use the sensitivity model formula
        # P(accept) = sigmoid(alpha_0 - beta_spread * max(0, rate - market_rate) - beta_burden * (loan/income) + beta_match * (loan/funded) + noise)

        grade = row['grade']
        market_rate = MARKET_RATE_BY_GRADE.get(grade, 0.24)
        loan = row['loan_amnt']
        income = row['annual_inc']
        funded = row['funded_amnt']

        spread = max(0, rate - market_rate)
        burden = loan / (income + 1)
        match = loan / (funded + 1)

        logit = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
        p_accept = 1.0 / (1.0 + np.exp(-logit))

        if p_accept < 0.001:
            continue

        expected_principal = p_accept * loan
        expected_interest = p_accept * loan * rate * (observed_time / 12)
        expected_loss = p_accept * loan * event
        expected_pnl = expected_interest - expected_loss

        if expected_pnl > best_pnl:
            best_pnl = expected_pnl
            best_rate = rate

    return best_rate, best_pnl

# Compute optimal rates for sample
print("Computing optimal rates for sample...")
sample_results = []
for i, row in sample_data.iterrows():
    observed_time = val_df.loc[val_df.index[i], 'observed_time']
    event = val_df.loc[val_df.index[i], 'event']
    opt_rate, opt_pnl = find_optimal_rate(row, final_pred[sample_idx[i]], observed_time, event)
    sample_results.append({'idx': i, 'opt_rate': opt_rate, 'opt_pnl': opt_pnl})

sample_df = pd.DataFrame(sample_results)
print(f"Sample optimal rate stats:")
print(sample_df['opt_rate'].describe())

# Now apply optimal rates to full dataset
print("Applying to full dataset...")
val_for_model['default_prob'] = final_pred

# For efficiency, bin by grade and compute average optimal rate per bin
# Actually, let's use a simpler approach: for each row, compute the optimal rate
# using the analytical formula from the sensitivity model

def compute_analytical_opt_rate(row, default_prob, observed_time, event):
    """Compute the rate that maximizes expected_pnl analytically."""
    grade = row['grade']
    market_rate = MARKET_RATE_BY_GRADE.get(grade, 0.24)
    loan = row['loan_amnt']
    income = row['annual_inc']
    funded = row['funded_amnt']

    burden = loan / (income + 1)
    match = loan / (funded + 1)

    best_pnl = -np.inf
    best_rate = 0.21

    # Grid search over rates
    for rate_cents in np.arange(21.0, 36.5, 0.5):
        rate = rate_cents / 100.0
        spread = max(0, rate - market_rate)

        logit = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
        if logit < -20:
            p_accept = 0
        else:
            p_accept = 1.0 / (1.0 + np.exp(-logit))

        if p_accept < 0.001:
            continue

        expected_principal = p_accept * loan
        expected_interest = p_accept * loan * rate * (observed_time / 12)
        expected_loss = p_accept * loan * event
        expected_pnl = expected_interest - expected_loss

        if expected_pnl > best_pnl:
            best_pnl = expected_pnl
            best_rate = rate

    return best_rate

# Apply to full dataset (vectorized approximation)
print("Computing analytical optimal rates...")
# Sample for speed
np.random.seed(42)
n_sample = min(100000, len(val_for_model))
sample_idx2 = np.random.choice(len(val_for_model), n_sample, replace=False)

sample_rates = []
for i in sample_idx2:
    row = val_for_model.iloc[i]
    rate = compute_analytical_opt_rate(row, final_pred[i], val_df.loc[val_df.index[i], 'observed_time'], val_df.loc[val_df.index[i], 'event'])
    sample_rates.append(rate)

sample_rates = np.array(sample_rates)
print(f"Sample optimal rate stats:")
print(f"  Mean: {sample_rates.mean():.4f}")
print(f"  Median: {np.median(sample_rates):.4f}")
print(f"  Std: {sample_rates.std():.4f}")
print(f"  25th pct: {np.percentile(sample_rates, 25):.4f}")
print(f"  75th pct: {np.percentile(sample_rates, 75):.4f}")

# Use the median optimal rate as a simple strategy
median_rate = np.median(sample_rates)
print(f"\nUsing median rate: {median_rate:.4f}")

val_for_model['offered_rate'] = median_rate
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

total_principal = sens_input['expected_principal'].sum()
loans_funded = sens_input['p_accept'].sum()
total_pnl = sens_input['expected_pnl'].sum()
acceptance_rate = sens_input['p_accept'].mean()

print(f"Median rate strategy: PnL=${total_pnl:,.0f}, Principal=${total_principal:,.0f}, "
      f"Loans={loans_funded:.0f}, Acept={acceptance_rate:.4f}")

# Also try per-grade optimal rates
print("\n--- Per-grade optimal rates ---")
for grade in ['C', 'D', 'E', 'F', 'G']:
    sub = val_for_model[val_for_for_model['grade'] == grade]
    if len(sub) == 0:
        continue

    observed_times_sub = val_df.loc[sub.index, 'observed_time'].values
    events_sub = val_df.loc[sub.index, 'event'].values

    best_pnl = -np.inf
    best_rate = 0.21

    for rate_cents in np.arange(21.0, 36.5, 0.5):
        rate = rate_cents / 100.0
        market_rate = MARKET_RATE_BY_GRADE.get(grade, 0.24)

        loan = sub['loan_amnt'].values
        income = sub['annual_inc'].values
        funded = sub['funded_amnt'].values

        burden = loan / (income + 1)
        match = loan / (funded + 1)
        spread = np.maximum(0, rate - market_rate)

        logit = 0.20 - 12.0 * spread - 1.5 * burden + 0.5 * match
        p_accept = 1.0 / (1.0 + np.exp(-logit))
        p_accept = np.clip(p_accept, 0, 1)

        expected_principal = p_accept * loan
        expected_interest = p_accept * loan * rate * (observed_times_sub / 12)
        expected_loss = p_accept * loan * events_sub
        expected_pnl = (expected_interest - expected_loss).sum()

        if expected_pnl > best_pnl:
            best_pnl = expected_pnl
            best_rate = rate

    print(f"  Grade {grade}: optimal_rate={best_rate:.4f}, expected_pnl=${best_pnl:,.0f}")

print("\n=== Summary of all strategies ===")
for r in results_list:
    print(f"  {r['type']}={r['param']}: PnL=${r['pnl']:,.0f}, Principal=${r['principal']:,.0f}, "
          f"Loans={r['loans']:.0f}, Acept={r['acceptance_rate']:.4f}")

# Pick best
best_result = max(results_list, key=lambda x: x['pnl'])
print(f"\nBest risk_mult strategy: mult={best_result['param']}, PnL=${best_result['pnl']:,.0f}")

# Also compare with median rate strategy
if total_pnl > best_result['pnl']:
    print(f"Median rate strategy is better: PnL=${total_pnl:,.0f}")
