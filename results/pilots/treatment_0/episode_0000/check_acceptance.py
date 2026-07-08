
import numpy as np
from data_pipeline.sensitivity_model import SensitivityModel

model = SensitivityModel()

# Check acceptance rates more carefully with many samples
for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    market_rate = model.market_rates[grade]
    min_viable = 0.21
    
    # At market rate
    probs_market = [model.predict_proba(grade, market_rate, 15000, 65000, 15000) for _ in range(500)]
    
    # At min viable rate (21%)
    probs_mv = [model.predict_proba(grade, min_viable, 15000, 65000, 15000) for _ in range(500)]
    
    # At market + 10pp
    probs_m10 = [model.predict_proba(grade, market_rate + 0.10, 15000, 65000, 15000) for _ in range(500)]
    
    print(f"{grade}: market={market_rate:.1%}, @market={np.mean(probs_market):.1%}, @minViable={np.mean(probs_mv):.1%}, @+10pp={np.mean(probs_m10):.1%}")
