import os
# Search for SensitivityModel definition
for root, dirs, files in os.walk('/home/oliversnavy/repos/ai-lending'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                    if 'SensitivityModel' in content:
                        print(f"Found in: {path}")
                        # Print relevant lines
                        for i, line in enumerate(content.split('\n')):
                            if 'SensitivityModel' in line or 'class' in line.lower():
                                print(f"  Line {i+1}: {line.strip()}")
            except:
                pass