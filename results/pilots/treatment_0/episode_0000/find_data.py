
import os
print("Current working directory:", os.getcwd())
print("\nContents of current dir:", os.listdir('.'))
print("\nContents of parent dir:", os.listdir('..'))

# Check if data exists
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if 'parquet' in f or f.endswith('.parquet'):
            print(os.path.join(root, f))
