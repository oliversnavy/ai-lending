
import json

results = {
    "pnl": 19676902.0,
    "c_stat": 0.6801,
    "acceptance_rate": 0.5174,
    "loans_funded": 705,
    "total_principal": 14989925.0,
    "approach": "Trained a logistic regression default risk model (ROC AUC 0.68) on 466k training loans, then designed grade-based pricing (C:21%, D:24%, E:29.5%, F:34%) with risk and burden adjustments. Used the sensitivity model to compute acceptance probabilities and greedily selected highest-ROI loans under $15M capital cap.",
    "hypothesis": "Higher-grade (E/F) borrowers with rates at or near market would accept at higher rates while still generating positive P&L due to high interest margins relative to expected losses."
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("results.json written:")
print(json.dumps(results, indent=2))
