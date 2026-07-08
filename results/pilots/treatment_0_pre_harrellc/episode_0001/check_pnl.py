import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')

# Check observed_time distribution
print("observed_time stats:")
print(val['observed_time'].describe())

# Check for a sample loan
sample = val[(val['grade']=='D') & (val['observed_time'] > 0)].head(5)
for _, row in sample.iterrows():
    interest = row['loan_amnt'] * 0.21 * (row['observed_time'] / 12)
    loss = row['loan_amnt'] * row['event']
    print(f"\nLoan ${row['loan_amnt']:.0f}, term={row['term']}, obs_time={row['observed_time']:.1f}mo")
    print(f"  Interest at 21%: ${interest:,.0f}")
    print(f"  Loss (event={row['event']}): ${loss:,.0f}")
    print(f"  P&L: ${interest - loss:,.0f}")
    print(f"  P&L / principal: {(interest - loss) / row['loan_amnt']:.4f}")

# Check median observed time by grade
print("\nMedian observed_time by grade:")
print(val.groupby('grade')['observed_time'].median())

# Check the actual P&L calculation
viable_grades = ['C', 'D', 'E', 'F']
val_v = val[val['grade'].isin(viable_grades)].copy()

# Offer at 21%
val_v['offered_rate'] = 0.21
market_rates = {'A': 0.095, 'B': 0.14, 'C': 0.19, 'D': 0.24, 'E': 0.295, 'F': 0.34, 'G': 0.37}
market_rate = val_v['grade'].map(market_rates).values
rate_spread = np.maximum(0, val_v['offered_rate'].values - market_rate)
burden = np.clip(val_v['loan_amnt'].values / np.maximum(val_v['annual_inc'].values, 1.0), 0, 10)
match = np.minimum(val_v['loan_amnt'].values / np.maximum(val_v['funded_amnt'].values, 1.0), 1.0)
log_odds = 0.20 - 12.0 * rate_spread - 1.5 * burden + 0.5 * match
p_accept = 1.0 / (1.0 + np.exp(-log_odds))
val_v['p_accept'] = p_accept

# P&L
val_v['expected_principal'] = val_v['p_accept'] * val_v['loan_amnt']
val_v['expected_interest'] = val_v['p_accept'] * val_v['loan_amnt'] * 0.21 * (val_v['observed_time'] / 12.0)
val_v['expected_loss'] = val_v['p_accept'] * val_v['loan_amnt'] * val_v['event']
val_v['expected_pnl'] = val_v['expected_interest'] - val_v['expected_loss']

# Check P&L per loan for a sample
print("\nSample P&L per expected loan:")
sample_pnl = val_v[['loan_amnt', 'observed_time', 'event', 'p_accept', 'expected_pnl']].head(20)
for _, row in sample_pnl.iterrows():
    print(f"  loan=${row['loan_amnt']:,.0f}, time={row['observed_time']:.1f}mo, event={row['event']}, p={row['p_accept']:.3f} → P&L=${row['expected_pnl']:,.0f}")

print(f"\nOverall stats:")
print(f"  Mean expected_principal per loan: ${val_v['expected_principal'].mean():,.0f}")
print(f"  Mean expected_pnl per loan: ${val_v['expected_pnl'].mean():,.0f}")
print(f"  Mean expected_interest per loan: ${val_v['expected_interest'].mean():,.0f}")
print(f"  Mean expected_loss per loan: ${val_v['expected_loss'].mean():,.0f}")
print(f"  Total expected principal: ${val_v['expected_principal'].sum():,.0f}")
print(f"  Total expected P&L: ${val_v['expected_pnl'].sum():,.0f}")