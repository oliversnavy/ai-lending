import pandas as pd
import numpy as np

train = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/train.parquet')

# Grade distribution and event rates
print("Grade distribution:")
print(train['grade'].value_counts())

print("\nEvent rate by grade:")
grade_stats = train.groupby('grade').agg(
    count=('event', 'count'),
    event_rate=('event', 'mean'),
    mean_int_rate=('int_rate', 'mean'),
    mean_loan_amnt=('loan_amnt', 'mean'),
    mean_annual_inc=('annual_inc', 'mean'),
    mean_dti=('dti', 'mean')
)
print(grade_stats)

print("\nSub-grade distribution (sample):")
print(train['sub_grade'].value_counts().head(20))

print("\nTerm distribution:")
print(train['term'].value_counts())

print("\nObserved time stats:")
print(train['observed_time'].describe())

print("\nEvent distribution:")
print(train['event'].value_counts())