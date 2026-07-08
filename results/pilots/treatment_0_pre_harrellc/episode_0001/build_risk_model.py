import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Preprocess term column
train['term_months'] = train['term'].str.extract('(\d+)').astype(float)
val['term_months'] = val['term'].str.extract('(\d+)').astype(float)

# Preprocess emp_length - extract numeric years
def emp_to_years(emp_str):
    if pd.isna(emp_str):
        return np.nan
    emp_str = str(emp_str).strip()
    if emp_str == '< 1 year':
        return 0.5
    if emp_str == '10+ years':
        return 10.0
    try:
        return float(emp_str.split()[0])
    except:
        return np.nan

train['emp_years'] = train['emp_length'].apply(emp_to_years)
val['emp_years'] = val['emp_length'].apply(emp_to_years)

# Feature engineering
for df in [train, val]:
    df['loan_to_inc'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    df['funded_to_requested'] = df['funded_amnt'] / df['loan_amnt'].clip(lower=1)
    df['dti_ratio'] = df['dti'] / 100.0
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    # Encode grade as ordinal
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_ord'] = df['grade'].map(grade_map)

# Define features
numeric_features = [
    'loan_amnt', 'funded_amnt', 'term_months', 'int_rate', 
    'fico_range_low', 'fico_range_high', 'fico_mid', 'dti', 'annual_inc', 
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 
    'open_acc', 'loan_to_inc', 'funded_to_requested', 'dti_ratio',
    'emp_years', 'grade_ord'
]

cat_features = ['sub_grade', 'emp_length', 'home_ownership', 
                'verification_status', 'purpose']

# Filter to rows with valid data
df_train = train.dropna(subset=['annual_inc', 'loan_amnt', 'funded_amnt', 'event']).copy()
df_val = val.dropna(subset=['annual_inc', 'loan_amnt', 'funded_amnt', 'event']).copy()

print(f"Train: {df_train.shape}, Val: {df_val.shape}")

# Build preprocessor
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Transform
X_train = preprocessor.fit_transform(df_train)
X_val = preprocessor.transform(df_val)
y_train = df_train['event'].values
y_val = df_val['event'].values

print(f"Feature matrix shape: {X_train.shape}")
print(f"Target distribution - train: {y_train.mean():.4f}, val: {y_val.mean():.4f}")

# Train risk model
risk_model = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
risk_model.fit(X_train, y_train)

# Evaluate
val_proba = risk_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_proba)
print(f"\nRisk Model AUC: {auc:.4f}")

# Save model
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'wb') as f:
    pickle.dump({'model': risk_model, 'preprocessor': preprocessor}, f)

print("Risk model saved.")