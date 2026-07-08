# Let me build the full pipeline step by step
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

# Load data
val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Load risk model
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
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
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

# Save val for later use
val.to_parquet('val_scored.parquet')
print("Val scored shape:", val.shape)
print("Score range:", val['default_prob'].min(), "-", val['default_prob'].max())

# Focus on grades C, D, E, F, G (our viable segment)
viable = val[val['grade'].isin(['C', 'D', 'E', 'F', 'G'])].copy()
print(f"\nViable segment: {len(viable)} applicants")
print(f"Grade distribution in viable:")
print(viable['grade'].value_counts())
print(f"\nDefault probs by grade:")
for g in ['C', 'D', 'E', 'F', 'G']:
    sub = viable[viable['grade'] == g]
    print(f"  {g}: default_prob mean={sub['default_prob'].mean():.4f}, median={sub['default_prob'].median():.4f}")
