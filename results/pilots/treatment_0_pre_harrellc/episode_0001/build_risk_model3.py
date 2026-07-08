import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Preprocess
train['term_months'] = train['term'].str.extract(r'(\d+)').astype(float)
val['term_months'] = val['term'].str.extract(r'(\d+)').astype(float)

def emp_to_years(emp_str):
    if pd.isna(emp_str): return np.nan
    emp_str = str(emp_str).strip()
    if emp_str == '< 1 year': return 0.5
    if emp_str == '10+ years': return 10.0
    try: return float(emp_str.split()[0])
    except: return np.nan

train['emp_years'] = train['emp_length'].apply(emp_to_years)
val['emp_years'] = val['emp_length'].apply(emp_to_years)

for df in [train, val]:
    df['loan_to_inc'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)
    df['funded_to_requested'] = df['funded_amnt'] / df['loan_amnt'].clip(lower=1)
    df['dti_ratio'] = df['dti'] / 100.0
    df['fico_mid'] = (df['fico_range_low'] + df['fico_range_high']) / 2.0
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_ord'] = df['grade'].map(grade_map)

numeric_features = [
    'loan_amnt', 'funded_amnt', 'term_months', 'int_rate', 
    'fico_range_low', 'fico_range_high', 'fico_mid', 'dti', 'annual_inc', 
    'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 
    'open_acc', 'loan_to_inc', 'funded_to_requested', 'dti_ratio',
    'emp_years', 'grade_ord'
]
cat_features = ['sub_grade', 'emp_length', 'home_ownership', 
                'verification_status', 'purpose']

df_train = train.dropna(subset=['annual_inc', 'loan_amnt', 'funded_amnt', 'event']).copy()
df_val = val.dropna(subset=['annual_inc', 'loan_amnt', 'funded_amnt', 'event']).copy()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, cat_features)
    ])

X_train = preprocessor.fit_transform(df_train)
X_val = preprocessor.transform(df_val)
y_train = df_train['event'].values
y_val = df_val['event'].values

risk_model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1,
    l2_regularization=0.1, min_samples_leaf=50
)
risk_model.fit(X_train, y_train)

val_proba = risk_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_proba)
print(f"Risk Model AUC: {auc:.4f}")

# Save
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001/risk_model.pkl', 'wb') as f:
    pickle.dump({'model': risk_model, 'preprocessor': preprocessor}, f)

print("Risk model saved.")
print(f"Val risk score stats: mean={val_proba.mean():.4f}, std={val_proba.std():.4f}, min={val_proba.min():.4f}, max={val_proba.max():.4f}")