# Vectorized pipeline - batch process sensitivity model calls
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/oliversnavy/repos/ai-lending')
from data_pipeline.sensitivity_model import SensitivityModel
import pickle
from sklearn.metrics import roc_auc_score

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
with open('/home/oliversnavy/repos/ai-lending/data/processed/sensitivity_model.pkl', 'rb') as f:
    sens_model = pickle.load(f)
with open('risk_model.pkl', 'rb') as f:
    saved = pickle.load(f)
model = saved['model']
feature_cols = saved['feature_cols']

def build_features(df):
    df = df.copy()
    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['grade_enc'] = df['grade'].map(grade_map)
    def sub_grade_to_num(sg):
        if pd.isna(sg): return 0
        g = sg[0]; i = int(sg[1:])
        return grade_map.get(g, 0) * 10 + i
    df['sub_grade_enc'] = df['sub_grade'].apply(sub_grade_to_num)
    emp_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 9, '10+ years': 10}
    df['emp_length_enc'] = df['emp_length'].map(emp_map).fillna(0).astype(float)
    home_map = {'ANY': 0, 'OTHER': 1, 'NONE': 2, 'MORTGAGE': 3, 'FORECLOSEURE': 4, 'OWN': 5}
    df['home_ownership_enc'] = df['home_ownership'].map(home_map).fillna(1).astype(float)
    purpose_map = {'credit_card': 0, 'debt_consolidation': 1, 'home_improvement': 2,
                   'major_purchase': 3, 'medical': 4, 'small_business': 5,
                   'vacation': 6, 'moving': 7, 'renewable_energy': 8,
                   'wedding': 9, 'house': 10, 'other': 11}
    df['purpose_enc'] = df['purpose'].map(purpose_map).fillna(11).astype(float)
    verif_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
    df['verification_status_enc'] = df['verification_status'].map(verif_map).fillna(0).astype(float)
    state_map = {
        'AL':0,'AK':1,'AZ':2,'AR':3,'CA':4,'CO':5,'CT':6,'DE':7,'FL':8,'GA':9,
        'HI':10,'ID':11,'IL':12,'IN':13,'IA':14,'KS':15,'KY':16,'LA':17,'ME':18,
        'MD':19,'MA':20,'MI':21,'MN':22,'MS':23,'MO':24,'MT':25,'NE':26,'NV':27,
        'NH':28,'NJ':29,'NM':30,'NY':31,'NC':32,'ND':33,'OH':34,'OK':35,'OR':36,
        'PA':37,'RI':38,'SC':39,'SD':40,'TN':41,'TX':42,'UT':43,'VT':44,'VA':45,
        'WA':46,'WV':47,'WI':48,'WY':49
    }
    df['state_code'] = df['addr_state'].map(state_map).fillna(50).astype(float)
    app_map = {'Individual': 0, 'Co-App': 1}
    df['application_type_enc'] = df['application_type'].map(app_map).fillna(0).astype(float)
    init_map = {'W': 0, 'F': 1}
    df['initial_list_status_enc'] = df['initial_list_status'].map(init_map).fillna(1).astype(float)
    df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float).fillna(36)
    X = df[feature_cols].fillna(0).astype(float)
    return X

X_val = build_features(val)
val_scores = model.predict_proba(X_val)[:, 1]
val['default_prob'] = val_scores

# Filter to viable segment
val = val[val['grade'].isin(['C', 'D', 'E', 'F', 'G'])].copy()
print(f"Viable segment: {len(val)} applicants")

# Create offered rate column - vectorized
# Base rate by grade
val['base_rate'] = val['grade'].map({
    'C': 0.25, 'D': 0.30, 'E': 0.35, 'F': 0.40, 'G': 0.40
})

# Risk adjustment
val['offered_rate'] = val['base_rate'] + (val['default_prob'] - 0.25) * 0.5
val['offered_rate'] = val['offered_rate'].clip(0.21, 0.50)

print("Offered rate stats:")
print(val['offered_rate'].describe())

# Batch process sensitivity model
print("\nBatch processing sensitivity model...")
BATCH_SIZE = 50000
n = len(val)
all_accept_probs = []

for i in range(0, n, BATCH_SIZE):
    batch = val.iloc[i:i+BATCH_SIZE]
    batch_df = pd.DataFrame([{
        'grade': g, 'offered_rate': r,
        'loan_amnt': la, 'annual_inc': ai, 'funded_amnt': fa
    } for g, r, la, ai, fa in zip(batch['grade'], batch['offered_rate'], 
                                    batch['loan_amnt'], batch['annual_inc'], 
                                    batch['funded_amnt'])])
    
    probs = sens_model.predict_proba_batch(batch_df)
    all_accept_probs.extend(probs)
    
    if (i + BATCH_SIZE) % 100000 == 0:
        print(f"  Processed {i + BATCH_SIZE}/{n}")

val['accept_prob'] = all_accept_probs

# Compute expected P&L
val['observed_time'] = val['observed_time'].fillna(28)
val['interest_yield'] = val['offered_rate'] * val['observed_time'] / 12 * (1 - val['default_prob'])
val['expected_pnl_per_dollar'] = val['interest_yield'] - val['default_prob']
val['weighted_pnl'] = val['accept_prob'] * val['expected_pnl_per_dollar'] * val['loan_amnt']
val['weighted_principal'] = val['accept_prob'] * val['loan_amnt']

print(f"\nTotal applicants: {len(val)}")
print(f"Positive expected P&L: {(val['weighted_pnl'] > 0).sum()}")
print(f"Total weighted P&L: ${val['weighted_pnl'].sum():,.0f}")
print(f"Total weighted principal: ${val['weighted_principal'].sum():,.0f}")

# Sort by weighted P&L and apply capital cap
val = val.sort_values('weighted_pnl', ascending=False)
cumulative = val['weighted_principal'].cumsum()
cap_mask = cumulative <= 15_000_000
selected = val[cap_mask]

print(f"\nAfter capital cap: {len(selected)} loans")
print(f"Total principal: ${selected['weighted_principal'].sum():,.0f}")
print(f"Total P&L: ${selected['weighted_pnl'].sum():,.0f}")
print(f"Acceptance rate: {selected['accept_prob'].mean():.4f}")

# C-statistic
c_stat = roc_auc_score(val['event'].values, val['default_prob'].values)
print(f"C-statistic: {c_stat:.4f}")

# Save results
import json
results_json = {
    "pnl": float(selected['weighted_pnl'].sum()),
    "c_stat": float(c_stat),
    "acceptance_rate": float(selected['accept_prob'].mean()),
    "loans_funded": int(len(selected)),
    "total_principal": float(selected['weighted_principal'].sum()),
    "approach": "GradientBoosting default risk model with grade-based rate schedule, targeting Grade C-F segment where risk-adjusted returns are positive",
    "hypothesis": "Higher-risk borrowers at higher rates would generate best risk-adjusted returns due to higher acceptance rates"
}

with open('results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print("\nResults saved!")
print(json.dumps(results_json, indent=2))
