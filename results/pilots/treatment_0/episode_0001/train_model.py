import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Feature engineering function
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
    features['num_tl_120dpd_2m'] = df['num_tl_120dpd_2m']
    features['num_tl_30dpd'] = df['num_tl_30dpd']
    features['num_tl_90g_dpd_24m'] = df['num_tl_90g_dpd_24m']
    features['tax_liens'] = df['tax_liens']
    features['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies']
    features['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    features['dti_squared'] = df['dti'] ** 2
    features['revol_bal_x_dti'] = df['revol_bal'] * df['dti']
    features['inq_x_dti'] = df['inq_last_6mths'] * df['dti']
    features['fico_range_low'] = df['fico_range_low']
    features['fico_range_high'] = df['fico_range_high']
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
    features['total_debt_to_income'] = (df['revol_bal'] + df['loan_amnt']) / df['annual_inc'].clip(lower=1)
    features['high_dti'] = (df['dti'] > 20).astype(int)
    features['high_loan_to_income'] = (df['loan_amnt'] / df['annual_inc'].clip(lower=1) > 0.5).astype(int)
    features['recent_inquiries'] = (df['inq_last_6mths'] > 3).astype(int)
    features['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    features['has_public_record'] = (df['pub_rec'] > 0).astype(int)
    
    return features

train_features = engineer_features(train)
train_features = train_features.fillna(0)
y_train = train['event']

# Train GBC model
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=50,
    subsample=0.8,
    random_state=42
)

# Cross validation
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, train_features, y_train, cv=cv, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit on full data
model.fit(train_features, y_train)

# Feature importance
importances = pd.Series(model.feature_importances_, index=train_features.columns)
print("\nTop 15 features:")
print(importances.sort_values(ascending=False).head(15))

# Save model
import pickle
with open('default_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved.")