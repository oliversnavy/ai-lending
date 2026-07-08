import numpy as np
import pandas as pd
import pickle

# Reproduce the SensitivityModel class
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

# Test it
model = SensitivityModel()
test_df = pd.DataFrame({
    'grade': ['C', 'C', 'D', 'D', 'E', 'E', 'B', 'B', 'A', 'A'],
    'offered_rate': [0.21, 0.25, 0.21, 0.25, 0.21, 0.25, 0.21, 0.14, 0.21, 0.14],
    'loan_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    'annual_inc': [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000],
    'funded_amnt': [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
})

probs = model.predict_proba_batch(test_df)
for i, row in test_df.iterrows():
    print(f"Grade {row['grade']}, rate {row['offered_rate']:.0%}: acceptance prob = {probs[i]:.4f}")