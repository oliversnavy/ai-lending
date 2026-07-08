import numpy as np
import pandas as pd
import pickle
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

# === Reproduce SensitivityModel ===
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class SensitivityModel:
    def __init__(self, market_rates=None, min_viable_rate=0.21, 
                 alpha_0=0.20, beta_spread=12.0, beta_burden=1.5, 
                 beta_match=0.50, noise_std=0.30, random_seed=42):
        self.market_rates = market_rates or {
            "A": 0.095, "B": 0.140, "C": 0.190, "D": 0.240, 
            "E": 0.295, "F": 0.340, "G": 0.370
        }
        self.min_viable_rate = min_viable_rate
        self.alpha_0 = alpha_0
        self.beta_spread = beta_spread
        self.beta_burden = beta_burden
        self.beta_match = beta_match
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_seed)

    def predict_proba_batch(self, df):
        market_rates = df["grade"].map(self.market_rates).fillna(0.25).values
        rate_spread = np.maximum(0.0, df["offered_rate"].values - market_rates)
        burden = df["loan_amnt"].values / np.maximum(df["annual_inc"].values, 1.0)
        match = np.minimum(df["loan_amnt"].values / np.maximum(df["funded_amnt"].values, 1.0), 1.0)
        noise = self.rng.normal(0, self.noise_std, size=len(df))
        log_odds = self.alpha_0 - self.beta_spread * rate_spread - self.beta_burden * burden + self.beta_match * match + noise
        below_floor = df["offered_rate"].values < self.min_viable_rate
        probs = _sigmoid(log_odds)
        probs[below_floor] = 0.0
        return probs

# === Load data ===
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# === Feature engineering ===
def engineer_features(df):
    df = df.copy()
    features = pd.DataFrame()
    features['loan_amnt'] = df['loan_amnt']
    features['funded_amnt'] = df['funded_amnt']
    features['funded_amnt_ratio'] = df['funded_amnt'] / df['loan_amnt'].clip(lower=1)
    features['annual_inc'] = df['annual_inc']
    features['dti'] = df['dti']
    features['term'] = (df['term'] == '60 months').astype(int)
    features['inq_last_6mths'] = df['inq_last_6mths']
    features['delinq_2yrs'] = df['delinq_2yrs']
    features['pub_rec'] = df['pub_rec']
    features['revol_bal'] = df['revol_bal']
    features['revol_util'] = df['revol_util']
    features['open_acc'] = df['open_acc']
    features['total_acc'] = df['total_acc']
    features['acc_now_delinq'] = df['acc_now_delinq']
    features['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med']
    features['chargeoff_within_12_mths'] = df['chargeoff_within_12_mths']
    features['tax_liens'] = df['tax_liens']
    features['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies']
    features['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    features['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    features['grade_num'] = df['grade'].map(grade_map)
    
    sub_grade_num = pd.Series(df['sub_grade'].map(lambda x: int(x[1:]) if isinstance(x, str) and len(x) > 1 and x[0] in 'ABCDEFG' else 0))
    features['sub_grade_num'] = sub_grade_num
    
    home_map = {'OWN': 1, 'MORTGAGE': 2, 'RENT': 0, 'OTHER': -1, 'NONE': -2}
    features['home_ownership'] = df['home_ownership'].map(home_map).fillna(-1).astype(int)
    features['verified'] = (df['verification_status'] == 'Verified').astype(int)
    
    purpose_map = {
        'debt_consolidation': 1, 'credit_card': 2, 'home_improvement': 3,
        'major_purchase': 4, 'small_business': 5, 'car': 6,
        'medical': 7, 'other': 8, 'renewable_energy': 9,
        'educational': 10, 'moving': 11, 'house': 12,
        'wedding': 13, 'vacation': 14, 'business': 15
    }
    features['purpose_num'] = df['purpose'].map(purpose_map).fillna(0).astype(int)
    features['individual'] = (df['application_type'] == 'Individual').astype(int)
    features['high_dti'] = (df['dti'] > 20).astype(int)
    features['recent_inquiries'] = (df['inq_last_6mths'] > 3).astype(int)
    features['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    features['has_public_record'] = (df['pub_rec'] > 0).astype(int)
    
    return features

val_features = engineer_features(val)
val_features = val_features.fillna(0)

# Load risk model from working directory
with open('default_model.pkl', 'rb') as f:
    risk_model = pickle.load(f)

# Score validation
val_risk_score = risk_model.predict_proba(val_features)[:, 1]

# Concordance index
c_index = concordance_index(val['observed_time'], -val_risk_score, val['event'])
auc = roc_auc_score(val['event'], val_risk_score)
print(f"Concordance index: {c_index:.4f}")
print(f"Validation AUC: {auc:.4f}")

# === Create acceptance model ===
sensitivity_model = SensitivityModel()

# === Pricing Strategy ===
# For each applicant, we need to set an offered rate.
# The rate should be: max(0.21, market_rate + risk_premium)
# where risk_premium scales with risk_score

# Key insight: We want to maximize expected P&L
# E[P&L] = P(accept) * [rate * loan_amnt * time/12 - default_prob * loan_amnt]
# 
# For a given applicant:
# - Higher rate -> lower P(accept) but higher margin if accepted
# - Lower rate -> higher P(accept) but lower margin
# - Optimal rate balances these

# Strategy: offer rate = max(0.21, market_rate + k * risk_score)
# where k is a tuning parameter

# Let me first understand the acceptance function better
# For each grade, find the rate that maximizes expected P&L

# For simplicity, let's use a fixed rate per grade for now
# Then optimize

print("\n=== Testing pricing strategy ===")
print("Testing different rate multipliers...")

for k in [0.0, 0.1, 0.2, 0.5, 1.0]:
    val_copy = val.copy()
    val_copy['risk_score'] = val_risk_score
    
    # Set offered rate
    market_rates = val['grade'].map(sensitivity_model.market_rates).values
    offered_rates = np.maximum(0.21, market_rates + k * val_risk_score)
    # Cap at 50%
    offered_rates = np.minimum(offered_rates, 0.50)
    val_copy['offered_rate'] = offered_rates
    
    # Get acceptance probabilities
    accept_probs = sensitivity_model.predict_proba_batch(val_copy)
    
    # Expected P&L per loan (simplified)
    # interest = loan_amnt * rate * avg_time/12
    # loss = loan_amnt * default_prob
    avg_time = val['observed_time'].mean()
    
    val_copy['accept_prob'] = accept_probs
    val_copy['expected_interest'] = val_copy['loan_amnt'] * val_copy['offered_rate'] * (avg_time / 12)
    val_copy['expected_loss'] = val_copy['loan_amnt'] * val_risk_score
    
    # Expected P&L per loan (weighted by acceptance)
    val_copy['expected_pnl'] = val_copy['accept_prob'] * (val_copy['expected_interest'] - val_copy['expected_loss'])
    
    total_pnl = val_copy['expected_pnl'].sum()
    total_principal = (val_copy['loan_amnt'] * val_copy['accept_prob']).sum()
    accept_rate = val_copy['accept_prob'].mean()
    count_accepted = (val_copy['accept_prob'] > 0).sum()
    
    print(f"k={k:.1f}: P&L=${total_pnl:,.0f}, Principal=${total_principal:,.0f}, Accept={accept_rate:.2%}, Count={count_accepted}")