import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Feature engineering function
def engineer_features(df):
    df = df.copy()
    
    # Basic features
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
    features['num_tl_120dpd_2m'] = df['num_tl_120dpd_2m']
    features['num_tl_30dpd'] = df['num_tl_30dpd']
    features['num_tl_90g_dpd_24m'] = df['num_tl_90g_dpd_24m']
    features['tax_liens'] = df['tax_liens']
    features['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies']
    
    # Income ratios
    features['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    features['dti_squared'] = features['dti'] ** 2
    features['revol_bal_x_dti'] = features['revol_bal'] * features['dti']
    features['inq_x_dti'] = features['inq_last_6mths'] * features['dti']
    
    # FICO range
    features['fico_range_low'] = df['fico_range_low']
    features['fico_range_high'] = df['fico_range_high']
    features['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    # Encode grade and sub_grade
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    features['grade_num'] = df['grade'].map(grade_map)
    
    # Sub-grade numeric
    sub_grade_num = pd.Series(df['sub_grade'].map(lambda x: int(x[1:]) if isinstance(x, str) and len(x) > 1 and x[0] in 'ABCDEFG' else 0))
    features['sub_grade_num'] = sub_grade_num
    
    # Home ownership encoding
    home_map = {'OWN': 1, 'MORTGAGE': 2, 'RENT': 0, 'OTHER': -1, 'NONE': -2}
    features['home_ownership'] = df['home_ownership'].map(home_map).fillna(-1).astype(int)
    
    # Verification status
    features['verified'] = (df['verification_status'] == 'Verified').astype(int)
    
    # Purpose encoding (one-hot for key categories)
    purpose_map = {
        'debt_consolidation': 1, 'credit_card': 2, 'home_improvement': 3,
        'major_purchase': 4, 'small_business': 5, 'car': 6,
        'medical': 7, 'other': 8, 'renewable_energy': 9,
        'educational': 10, 'moving': 11, 'house': 12,
        'wedding': 13, 'vacation': 14, 'business': 15
    }
    features['purpose_num'] = df['purpose'].map(purpose_map).fillna(0).astype(int)
    
    # Application type
    features['individual'] = (df['application_type'] == 'Individual').astype(int)
    
    # Derived features
    features['total_debt_to_income'] = (features['revol_bal'] + features['loan_amnt']) / df['annual_inc'].clip(lower=1)
    features['high_dti'] = (features['dti'] > 20).astype(int)
    features['high_loan_to_income'] = (features['loan_to_income'] > 0.5).astype(int)
    features['recent_inquiries'] = (features['inq_last_6mths'] > 3).astype(int)
    features['has_delinquency'] = (features['delinq_2yrs'] > 0).astype(int)
    features['has_public_record'] = (features['pub_rec'] > 0).astype(int)
    
    return features

# Engineer features for train
train_features = engineer_features(train)

# Target: event (1 = charged off)
y_train = train['event']

# Check for NaN
print("NaN counts:")
print(train_features.isnull().sum()[train_features.isnull().sum() > 0])

# Fill remaining NaN
train_features = train_features.fillna(0)

print("\nFeature matrix shape:", train_features.shape)
print("Target distribution:", y_train.value_counts().to_dict())