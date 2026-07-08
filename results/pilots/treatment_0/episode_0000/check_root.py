import os
# Check the project root
root = '/home/oliversnavy/repos/ai-lending'
print("Files in project root:", os.listdir(root))
print("\nFiles in data:", os.listdir(os.path.join(root, 'data')) if os.path.exists(os.path.join(root, 'data')) else "NO DATA")
print("\nFiles in data/processed:", os.listdir(os.path.join(root, 'data/processed')) if os.path.exists(os.path.join(root, 'data/processed')) else "NO PROCESSED")