import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

with open('risk_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoders']
feature_cols = model_data['feature_cols']
grade_order = model_data['grade_order']

# Load validation data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Select ALL columns from val that we need
val_cols = ['loan_amnt', 'funded_amnt', 'int_rate', 'fico_range_low', 'fico_range_high', 'dti', 'annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'revol_bal', 'revol_util', 'open_acc', 'total_acc', 'mths_since_last_delinq', 'mths_since_last_record', 'collections_12_mths_ex_med', 'acc_now_delinq', 'pub_rec_bankruptcies', 'tax_liens', 'chargeoff_within_12_mths', 'delinq_amnt', 'grade', 'sub_grade', 'home_ownership', 'purpose', 'verification_status', 'emp_length', 'term']

val_enc = val[val_cols].copy()

# Create encoded columns
for col in ['grade', 'sub_grade', 'home_ownership', 'purpose', 'verification_status', 'emp_length']:
    if col in le:
        val_enc[f'{col}_enc'] = le[col].transform(val_enc[col].astype(str).fillna('NONE'))

val_enc['grade_num'] = val_enc['grade'].map(grade_order).fillna(3)
val_enc['term_enc'] = (val_enc['term'] == '60 months').astype(int)

# Select all needed columns
final_cols = [c for c in feature_cols if c in val_enc.columns]
print(f"Final cols: {final_cols}")

X_val = val_enc[final_cols].fillna(0).astype(float)

# Score
prob_default = model.predict_proba(X_val)[:, 1]
val['prob_default'] = prob_default

print(f"\nDefault probability stats:")
print(val['prob_default'].describe())

y_val = val['event'].astype(int)
print(f"\nDefault rate in val: {y_val.mean():.4f}")
print(f"Validation AUC: {roc_auc_score(y_val, prob_default):.4f}")

# Save for later
val.to_parquet('val_scored.parquet', index=False)
print("\nSaved val_scored.parquet")