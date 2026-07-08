import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')

import numpy as np
import pandas as pd
import pickle
from data_pipeline.sensitivity_model import SensitivityModel
from sklearn.metrics import roc_auc_score
import json

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
with open('risk_model.pkl', 'rb') as f:
    risk_model = pickle.load(f)

sens_model = SensitivityModel()

# Feature engineering function
def engineer_features(df):
    df = df.copy()
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_num'] = df['grade'].map(grade_map)
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['dti_ratio'] = df['dti'] / 100
    df['lti'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['term_months'] = df['term'].str.replace(' months', '').astype(float)
    df['income_log'] = np.log1p(df['annual_inc'])
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_inq'] = (df['inq_last_6mths'] > 0).astype(int)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    df['has_chargeoff'] = (df['chargeoff_within_12_mths'] > 0).astype(int)
    df['has_collections'] = (df['collections_12_mths_ex_med'] > 0).astype(int)
    
    df['sub_grade_num'] = df['sub_grade'].str[0].map(grade_map) + df['sub_grade'].str[1:].astype(float) / 10
    df['home_own_num'] = df['home_ownership'].map({'OTHER': 0, 'NONE': 1, 'MORTGAGE': 2, 'RENT': 3, 'OWN': 4})
    df['verify_num'] = df['verification_status'].map({'Not Verified': 0, 'Source Verified': 1, 'Verified': 2})
    df['purpose_num'] = df['purpose'].map({
        'car': 1, 'credit_card': 2, 'education': 3, 'home_improvement': 4,
        'major_purchase': 5, 'medical': 6, 'moving': 7, 'other': 8,
        'debt_consolidation': 9, 'renewable_credit': 10, 'wedding': 11
    })
    
    feature_cols = [
        'grade_num', 'sub_grade_num', 'fico_mid', 'fico_range_low', 'fico_range_high',
        'loan_amnt', 'funded_amnt', 'term_months', 'int_rate',
        'dti', 'dti_ratio', 'lti', 'annual_inc', 'income_log',
        'revol_bal', 'open_acc', 'total_acc',
        'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq',
        'collections_12_mths_ex_med', 'mths_since_last_major_derog',
        'has_delinq', 'has_inq', 'has_pub_rec',
        'has_chargeoff', 'has_collections',
        'home_own_num', 'verify_num', 'purpose_num',
        'pub_rec_bankruptcies', 'tax_liens',
        'mo_sin_rcnt_tl', 'mths_since_recent_bc',
        'num_accts_ever_120_pd', 'num_rev_accts', 'num_bc_tl',
        'pct_tl_nvr_dlq', 'percent_bc_gt_75',
        'inq_last_12m', 'acc_open_past_24mths',
        'total_bal_il', 'il_util', 'max_bal_bc',
        'bc_open_to_buy', 'bc_util',
        'tot_cur_bal', 'avg_cur_bal',
        'num_op_rev_tl', 'num_tl_op_past_12m',
        'num_rev_tl_bal_gt_0',
    ]
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    return df[existing_cols]

# Score all validation applicants
X_val = engineer_features(val).fillna(0).values.astype(float)
val['pred_prob'] = risk_model.predict_proba(X_val)[:, 1]

# Filter to viable grades
viable_grades = ['C', 'D', 'E', 'F', 'G']
val_viable = val[val['grade'].isin(viable_grades)].copy()

# Best config: base=0.22, mult=0.4
BASE_RATE = 0.22
RISK_MULT = 0.4

rates = np.clip(BASE_RATE + RISK_MULT * val_viable['pred_prob'].values, 0.21, 0.40)

offer_df = pd.DataFrame({
    'grade': val_viable['grade'].values,
    'offered_rate': rates,
    'loan_amnt': val_viable['loan_amnt'].values,
    'annual_inc': val_viable['annual_inc'].values,
    'funded_amnt': val_viable['funded_amnt'].values,
})
accept_probs = sens_model.predict_proba_batch(offer_df)

interest = val_viable['loan_amnt'] * rates * (val_viable['observed_time'] / 12)
expected_loss = val_viable['loan_amnt'] * val_viable['pred_prob']
expected_pnl = accept_probs * (interest - expected_loss)
expected_pnl_per_dollar = expected_pnl / val_viable['loan_amnt']

val_viable['accept_prob'] = accept_probs
val_viable['interest'] = interest
val_viable['exp_pnl'] = expected_pnl
val_viable['exp_pnl_per_dollar'] = expected_pnl_per_dollar
val_viable['offered_rate'] = rates

# Select positive P&L loans up to cap, ranked by P&L per dollar
positive = val_viable[val_viable['exp_pnl'] > 0].sort_values('exp_pnl_per_dollar', ascending=False)
positive['cum_principal'] = positive['loan_amnt'].cumsum()
selected = positive[positive['cum_principal'] <= 15_000_000]

# Compute metrics
c_stat = roc_auc_score(val['event'], val['pred_prob'])
total_pnl = selected['exp_pnl'].sum()
total_principal = selected['loan_amnt'].sum()
loans_funded = len(selected)
overall_accept = accept_probs.mean()

print(f"=== FINAL RESULTS ===")
print(f"Expected P&L: ${total_pnl:,.0f}")
print(f"C-stat: {c_stat:.4f}")
print(f"Acceptance rate: {overall_accept:.3f}")
print(f"Loans funded: {loans_funded}")
print(f"Total principal: ${total_principal:,.0f}")
print(f"P&L per dollar: {total_pnl / total_principal:.3f}")
print(f"Mean offered rate: {selected['offered_rate'].mean():.3f}")
print(f"Mean acceptance prob: {selected['accept_prob'].mean():.3f}")

# Grade breakdown
print(f"\nGrade breakdown:")
print(selected['grade'].value_counts())

# Rate distribution
print(f"\nRate distribution:")
print(selected['offered_rate'].describe())

# Write results.json
results = {
    "pnl": float(total_pnl),
    "c_stat": float(c_stat),
    "acceptance_rate": float(overall_accept),
    "loans_funded": int(loans_funded),
    "total_principal": float(total_principal),
    "approach": "Trained gradient boosting default risk model (AUC=0.70) on LendingClub training data, then optimized portfolio by offering risk-adjusted rates (22% base + 40% of default probability) to Grade C-F borrowers, ranking by expected P&L per dollar within $15M capital cap.",
    "hypothesis": "Risk-adjusted pricing that increases rates for higher-default-probability loans would maximize P&L by better compensating for expected losses while maintaining competitive acceptance rates."
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n=== results.json ===")
print(json.dumps(results, indent=2))