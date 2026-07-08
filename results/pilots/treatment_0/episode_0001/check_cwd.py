import os
# Check working directory
import subprocess
result = subprocess.run(['pwd'], capture_output=True, text=True, cwd='/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001')
print("CWD:", result.stdout)

# Check what's in the working directory
result2 = subprocess.run(['ls', '-la'], capture_output=True, text=True, cwd='/home/oliversnavy/repos/ai-lending/results/skills/treatment_0/episode_0001')
print("Files:", result2.stdout)