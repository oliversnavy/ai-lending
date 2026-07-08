import os
# Check directory structure
for root, dirs, files in os.walk('data'):
    print(f"DIR: {root}")
    for f in files[:10]:
        print(f"  FILE: {f}")