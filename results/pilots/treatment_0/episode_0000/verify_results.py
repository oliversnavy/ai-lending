
# Final verification - read the results.json file
import json
with open('/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0000/results.json', 'r') as f:
    results = json.load(f)

print("Final results.json content:")
print(json.dumps(results, indent=2))
