import json

# Re-read the file to verify
with open('results.json', 'r') as f:
    content = f.read()
    print("Current results.json content:")
    print(content)

# Verify it matches what we computed
data = json.loads(content)
print(f"\nVerification:")
print(f"  pnl: {data['pnl']}")
print(f"  c_stat: {data['c_stat']}")
print(f"  acceptance_rate: {data['acceptance_rate']}")
print(f"  loans_funded: {data['loans_funded']}")
print(f"  total_principal: {data['total_principal']}")
