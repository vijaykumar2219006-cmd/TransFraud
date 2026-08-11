import pandas as pd

file_path = r"datasets\PaySim\PS_20174392719_1491204439457_log.csv"

chunk_size = 500_000

fraud_amounts = []
normal_amounts = []

fraud_count_by_type = {}
normal_count_by_type = {}

for chunk in pd.read_csv(
    file_path,
    chunksize=chunk_size,
    usecols=["amount", "type", "isFraud"]
):
    
    fraud = chunk[chunk["isFraud"] == 1]
    normal = chunk[chunk["isFraud"] == 0]

    fraud_amounts.extend(fraud["amount"].tolist())
    normal_amounts.extend(normal["amount"].tolist())

    for t, count in fraud["type"].value_counts().items():
        fraud_count_by_type[t] = fraud_count_by_type.get(t, 0) + count

    for t, count in normal["type"].value_counts().items():
        normal_count_by_type[t] = normal_count_by_type.get(t, 0) + count


fraud_series = pd.Series(fraud_amounts)
normal_series = pd.Series(normal_amounts)

print("\n========== FRAUD AMOUNT ANALYSIS ==========")

print("\nFraud transactions:")
print("Count:", len(fraud_series))
print("Minimum:", fraud_series.min())
print("Maximum:", fraud_series.max())
print("Mean:", fraud_series.mean())
print("Median:", fraud_series.median())

print("\nFraud amount percentiles:")
print(fraud_series.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

print("\nLegitimate transactions:")
print("Count:", len(normal_series))
print("Minimum:", normal_series.min())
print("Maximum:", normal_series.max())
print("Mean:", normal_series.mean())
print("Median:", normal_series.median())

print("\n========== FRAUD BY TRANSACTION TYPE ==========")

for t, count in fraud_count_by_type.items():
    print(t, ":", count)

print("\n========== LEGITIMATE BY TRANSACTION TYPE ==========")

for t, count in normal_count_by_type.items():
    print(t, ":", count)