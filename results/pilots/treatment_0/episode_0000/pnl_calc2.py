
import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Calculate expected P&L for each grade at different rates
for grade in ['C', 'D', 'E', 'F', 'G']:
    subset = train[train['grade'] == grade]
    default_rate = subset['event'].mean()
    avg_time_nd = subset[subset['event'] == 0]['observed_time'].mean()
    avg_time_d = subset[subset['event'] == 1]['observed_time'].mean()
    
    print(f"\nGrade {grade}: default_rate={default_rate:.4f}, avg_time_nd={avg_time_nd:.1f}, avg_time_d={avg_time_d:.1f}")
    
    for rate in [0.21, 0.25, 0.30, 0.35, 0.40]:
        # Expected interest from non-defaulters
        interest_nd = 15000 * rate * avg_time_nd / 12
        # Expected interest from defaulters
        interest_d = 15000 * rate * avg_time_d / 12
        # Total expected interest
        exp_interest = (1 - default_rate) * interest_nd + default_rate * interest_d
        # Expected loss
        exp_loss = default_rate * 15000
        # Net P&L
        pnl = exp_interest - exp_loss
        print(f"  Rate {rate:.0%}: P&L = ${pnl:,.0f}")
