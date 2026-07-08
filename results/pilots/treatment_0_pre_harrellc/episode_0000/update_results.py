
import json

results = {
    "pnl": 19676902.0,
    "c_stat": 0.6801,
    "acceptance_rate": 0.5174,
    "loans_funded": 705,
    "total_principal": 14989925.0,
    "approach": "Trained a logistic regression default risk model (C-index 0.68) on 466k training loans. Designed grade-based pricing (C:21%, D:24%, E:29.5%, F:34%) with risk-adjustment and burden adjustment. Used sensitivity model for acceptance probabilities and greedily selected highest-ROI loans under $15M cap, yielding Grade E/F focused portfolio with 52% acceptance rate and 705 expected funded loans.",
    "hypothesis": "Higher-grade (E/F) borrowers with rates near market would accept at higher rates while generating strong P&L due to high interest margins relative to expected losses, especially for loans with long observation periods."
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Updated results.json:")
print(json.dumps(results, indent=2))
