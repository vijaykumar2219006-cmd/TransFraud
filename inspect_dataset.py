import pandas as pd

file_path = r"datasets\PaySim\PS_20174392719_1491204439457_log.csv"

chunk_size = 500_000

total_rows = 0
fraud_count = 0
legitimate_count = 0

unique_senders = set()
unique_receivers = set()

type_counts = {}
fraud_by_type = {}

missing_values = None

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    total_rows += len(chunk)

    # Fraud / legitimate
    fraud_count += (chunk["isFraud"] == 1).sum()
    legitimate_count += (chunk["isFraud"] == 0).sum()

    # Unique sender / receiver
    unique_senders.update(chunk["nameOrig"].unique())
    unique_receivers.update(chunk["nameDest"].unique())

    # Transaction type counts
    for transaction_type, count in chunk["type"].value_counts().items():
        type_counts[transaction_type] = (
            type_counts.get(transaction_type, 0) + count
        )

    # Fraud by transaction type
    fraud_chunk = chunk[chunk["isFraud"] == 1]

    for transaction_type, count in fraud_chunk["type"].value_counts().items():
        fraud_by_type[transaction_type] = (
            fraud_by_type.get(transaction_type, 0) + count
        )

    # Missing values
    chunk_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = chunk_missing
    else:
        missing_values += chunk_missing


print("\n========== DATASET SUMMARY ==========")

print("Total transactions:", total_rows)
print("Fraud transactions:", fraud_count)
print("Legitimate transactions:", legitimate_count)

fraud_percentage = (fraud_count / total_rows) * 100

print("Fraud percentage:", fraud_percentage)

print("\nUnique senders:", len(unique_senders))
print("Unique receivers:", len(unique_receivers))

print("\nTransaction types:")
for key, value in type_counts.items():
    print(key, ":", value)

print("\nFraud by transaction type:")
for key, value in fraud_by_type.items():
    print(key, ":", value)

print("\nMissing values:")
print(missing_values)