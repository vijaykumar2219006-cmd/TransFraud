import pandas as pd

file_path = r"datasets\SyntheticBanking\Fraud Detection Dataset.csv"

# Read only the first 10 rows
df = pd.read_csv(file_path, nrows=10)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST 10 ROWS ==========")
print(df)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== SHAPE OF SAMPLE ==========")
print(df.shape)