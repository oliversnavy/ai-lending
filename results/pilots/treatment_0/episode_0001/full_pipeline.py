import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending/data/processed')
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
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

    def predict_proba(self, grade, offered_rate, loan_amnt, annual_inc, funded_amnt):
        if offered_rate < self.min_viable_rate:
            return 0.0
        market_rate = self.market_rates.get(grade, 0.25)
        rate_spread = max(0.0, offered_rate - market_rate)
        burden = loan_amnt / max(annual_inc, 1.0)
        match = min(loan_amnt / max(funded_amnt, 1.0), 1.0)
        noise = self.rng.normal(0, self.noise_std)
        log_odds = self.alpha_0 - self.beta_spread * rate_spread - self.beta_burden * burden + self.beta_match * match + noise
        return float(_sigmoid(np.array(log_odds)))

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

train_features = engineer_features(train)
train_features = train_features.fillna(0)
val_features = engineer_features(val)
val_features = val_features.fillna(0)

# === Load risk model ===
with open('/home/oliversnavy/repos/ai-lending/data/processed/default_model.pkl', 'rb') as f:
    risk_model = pickle.load(f)

# Score validation
val_risk_score = risk_model.predict_proba(val_features)[:, 1]

# Concordance index
c_index = concordance_index(val['observed_time'], -val_risk_score, val['event'])
auc = roc_auc_score(val['event'], val_risk_score)
print(f"Concordance index: {c_index:.4f}")
print(f"Validation AUC: {auc:.4f}")

# === Pricing strategy ===
# We need to offer rates >= 21% (floor) and price based on risk
# Higher risk = higher rate, but must stay competitive

# For each applicant, determine:
# 1. Their risk score (default probability)
# 2. Their grade (determines market rate)
# 3. Set offered rate = max(0.21, market_rate + risk_premium)

# Risk premium: higher risk -> higher premium
# But we need to keep acceptance probability reasonable

# Strategy: offer rate = max(MIN_VIABLE, market_rate + risk_adjustment)
# where risk_adjustment is based on risk score

# Let's try: offered_rate = max(0.21, market_rate + 0.05 * (risk_score - 0.15))
# This adds a small premium for risk above baseline

# Actually, let me think about this more carefully.
# We want to maximize P&L = interest_collected - principal_lost
# For a loan: E[P&L] = prob_accept * (loan_amnt * rate * time/12 - loan_amnt * default_prob)
# We need to find the rate that maximizes this.

# For simplicity, let's start with a fixed rate per grade and see what happens.
# Then iterate.

# First, let's see what rates we should offer by grade
sensitivity_model = SensitivityModel()

# Test different rates by grade
for grade in ['C', 'D', 'E', 'F', 'G']:
    market_rate = sensitivity_model.market_rates[grade]
    for rate_offset in [0, 0.02, 0.05, 0.10]:
        offered_rate = min(market_rate + rate_offset, 0.50)
        if offered_rate < 0.21:
            continue
        test_df = pd.DataFrame({
            'grade': [grade], 'offered_rate': [offered_rate],
            'loan_amnt': [15000], 'annual_inc': [60000], 'funded_amnt': [15000]
        })
        prob = sensitivity_model.predict_proba_batch(test_df)[0]
        print(f"Grade {grade}: market={market_rate:.1%}, offered={offered_rate:.1%}, acceptance={prob:.2%}")