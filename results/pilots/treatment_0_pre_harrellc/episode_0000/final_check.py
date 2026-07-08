
# Final verification - read the file as it exists on disk
import json
with open('results.json', 'r') as f:
    data = json.load(f)
    
print("Final verification:")
print(f"  P&L: ${data['pnl']:,.0f}")
print(f"  C-index: {data['c_stat']:.4f}")
print(f"  Acceptance rate: {data['acceptance_rate']:.4f}")
print(f"  Loans funded: {data['loans_funded']}")
print(f"  Total principal: ${data['total_principal']:,.0f}")
print(f"  All constraints met: {data['total_principal'] <= 15_000_000 and data['loans_funded'] >= 400 and data['acceptance_rate'] > 0.01}")
