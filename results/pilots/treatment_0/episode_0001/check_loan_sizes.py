import pandas as pd

val = pd.read_parquet('/home/oliversnavy/repos/ai-lending/data/processed/val.parquet')
print("Average loan_amnt:", val['loan_amnt'].mean())
print("Median loan_amnt:", val['loan_amnt'].median())
print("Loan amount distribution:")
print(val['loan_amnt'].describe())
print("\nBy grade:")
print(val.groupby('grade')['loan_amnt'].mean())
print("\nBy grade, count:")
print(val.groupby('grade')['loan_amnt'].count())
print("\nAverage term:")
print(val['term'].value_counts())