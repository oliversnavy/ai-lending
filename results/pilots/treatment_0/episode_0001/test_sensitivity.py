import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Feature engineering
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

# Load model
with open('default_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Score validation set
val_features = engineer_features(val)
val_features = val_features.fillna(0)
val_risk_score = model.predict_proba(val_features)[:, 1]

# Load sensitivity model for acceptance
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sensitivity_model = pickle.load(f)

# Now let's understand the sensitivity model better
# Test a few scenarios
test_cases = [
    {'grade': 'C', 'offered_rate': 0.21, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'C', 'offered_rate': 0.25, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'D', 'offered_rate': 0.21, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'D', 'offered_rate': 0.25, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'E', 'offered_rate': 0.21, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'E', 'offered_rate': 0.25, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'B', 'offered_rate': 0.21, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
    {'grade': 'B', 'offered_rate': 0.14, 'loan_amnt': 10000, 'annual_inc': 50000, 'funded_amnt': 10000},
]

for tc in test_cases:
    prob = sensitivity_model.predict_proba(**tc)
    print(f"Grade {tc['grade']}, rate {tc['offered_rate']:.0%}: acceptance prob = {float(prob):.4f}")