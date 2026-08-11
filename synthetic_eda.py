import pandas as pd
from collections import Counter

file_path = r"datasets\SyntheticBanking\Fraud Detection Dataset.csv"

chunk_size = 100_000

total_rows = 0
fraud_count = 0
legitimate_count = 0

customers = set()
cards = set()
devices = set()
merchants = set()

customer_counts = Counter()
merchant_counts = Counter()
card_counts = Counter()
device_counts = Counter()

customer_merchant_counts = Counter()

fraud_types = Counter()
transaction_types = Counter()

missing_values = None

for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    total_rows += len(chunk)

    # Fraud distribution
    fraud_count += (chunk["is_fraud"] == 1).sum()
    legitimate_count += (chunk["is_fraud"] == 0).sum()

    # Unique entities
    customers.update(chunk["customer_id"].dropna().unique())
    cards.update(chunk["card_id"].dropna().unique())
    devices.update(chunk["device_id"].dropna().unique())
    merchants.update(chunk["merchant_id"].dropna().unique())

    # Frequency
    customer_counts.update(chunk["customer_id"])
    merchant_counts.update(chunk["merchant_id"])
    card_counts.update(chunk["card_id"])
    device_counts.update(chunk["device_id"])

    # Customer → merchant relationship
    pairs = zip(
        chunk["customer_id"],
        chunk["merchant_id"]
    )
    customer_merchant_counts.update(pairs)

    # Transaction types
    transaction_types.update(chunk["transaction_type"])

    # Fraud types
    fraud_rows = chunk[chunk["is_fraud"] == 1]

    if "fraud_type" in fraud_rows.columns:
        fraud_types.update(
            fraud_rows["fraud_type"].dropna()
        )

    # Missing values
    current_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = current_missing
    else:
        missing_values += current_missing


print("\n========== DATASET SUMMARY ==========")

print("Total transactions:", total_rows)
print("Fraud transactions:", fraud_count)
print("Legitimate transactions:", legitimate_count)

print(
    "Fraud percentage:",
    (fraud_count / total_rows) * 100
)

print("\n========== UNIQUE ENTITIES ==========")

print("Unique customers:", len(customers))
print("Unique cards:", len(cards))
print("Unique devices:", len(devices))
print("Unique merchants:", len(merchants))

print("\n========== CUSTOMER HISTORY ==========")

print(
    "Customers with 1+ transactions:",
    len(customer_counts)
)

print(
    "Customers with 2+ transactions:",
    sum(x >= 2 for x in customer_counts.values())
)

print(
    "Customers with 5+ transactions:",
    sum(x >= 5 for x in customer_counts.values())
)

print(
    "Customers with 10+ transactions:",
    sum(x >= 10 for x in customer_counts.values())
)

print(
    "Customers with 50+ transactions:",
    sum(x >= 50 for x in customer_counts.values())
)

print("\n========== MERCHANT HISTORY ==========")

print(
    "Merchants with 2+ transactions:",
    sum(x >= 2 for x in merchant_counts.values())
)

print(
    "Merchants with 10+ transactions:",
    sum(x >= 10 for x in merchant_counts.values())
)

print(
    "Merchants with 100+ transactions:",
    sum(x >= 100 for x in merchant_counts.values())
)

print("\n========== CUSTOMER → MERCHANT ==========")

print(
    "Unique customer-merchant pairs:",
    len(customer_merchant_counts)
)

print(
    "Repeated customer-merchant pairs:",
    sum(
        x > 1
        for x in customer_merchant_counts.values()
    )
)

print(
    "Pairs with 5+ transactions:",
    sum(
        x >= 5
        for x in customer_merchant_counts.values()
    )
)

print("\n========== TRANSACTION TYPES ==========")

for key, value in transaction_types.items():
    print(key, ":", value)

print("\n========== FRAUD TYPES ==========")

for key, value in fraud_types.items():
    print(key, ":", value)

print("\n========== MISSING VALUES ==========")

print(missing_values)