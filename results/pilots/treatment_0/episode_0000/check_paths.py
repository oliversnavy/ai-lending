import os
# Check what's in the current directory and parent directories
print("CWD:", os.getcwd())
print("\nFiles in current dir:", os.listdir('.'))
print("\nFiles in parent dir:", os.listdir('..'))
print("\nFiles in data dir:", os.listdir('data') if os.path.exists('data') else "NO DATA DIR")
print("\nFiles in data/processed:", os.listdir('data/processed') if os.path.exists('data/processed') else "NO PROCESSED DIR")