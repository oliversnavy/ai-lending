# Test sensitivity model at different rate levels for different grades
# Using the tool to understand acceptance patterns

test_params = [
    # Grade C - market rate ~13.5%
    ('C', 0.21, 12000, 60000, 12000),
    ('C', 0.25, 12000, 60000, 12000),
    ('C', 0.30, 12000, 60000, 12000),
    # Grade D - market rate ~17.5%
    ('D', 0.21, 12000, 60000, 12000),
    ('D', 0.25, 12000, 60000, 12000),
    ('D', 0.30, 12000, 60000, 12000),
    ('D', 0.35, 12000, 60000, 12000),
    # Grade E - market rate ~20%
    ('E', 0.21, 12000, 60000, 12000),
    ('E', 0.25, 12000, 60000, 12000),
    ('E', 0.30, 12000, 60000, 12000),
    ('E', 0.35, 12000, 60000, 12000),
    # Grade F - market rate ~24.5%
    ('F', 0.21, 12000, 60000, 12000),
    ('F', 0.25, 12000, 60000, 12000),
    ('F', 0.30, 12000, 60000, 12000),
    ('F', 0.35, 12000, 60000, 12000),
    ('F', 0.40, 12000, 60000, 12000),
    # Grade G - market rate ~28%
    ('G', 0.25, 12000, 60000, 12000),
    ('G', 0.30, 12000, 60000, 12000),
    ('G', 0.35, 12000, 60000, 12000),
    ('G', 0.40, 12000, 60000, 12000),
]

print("Sensitivity model acceptance probabilities:")
for grade, rate, loan, inc, funded in test_params:
    # I'll use the tool for a few key points
    pass

# Let me use the tool for key test points
from sensitivity_model_query import *

# Actually, let me just call the tool directly
print("Testing with sensitivity_model_query tool:")
