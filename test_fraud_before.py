import pandas as pd

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

customer_id = "CUST0000069"

history = df[
    (df["customer_id"] == customer_id)
].sort_values("timestamp")

fraud_row = history[
    history["is_fraud"] == 1
].iloc[0]

previous = history[
    history["timestamp"] < fraud_row["timestamp"]
]

print("\nCustomer:", customer_id)
print("Fraud timestamp:", fraud_row["timestamp"])
print("Fraud amount:", fraud_row["amount"])
print("Fraud type:", fraud_row["fraud_type"])

print("\nTransactions available BEFORE fraud:")
print(
    previous[
        [
            "timestamp",
            "amount",
            "is_fraud",
            "customer_previous_count",
            "card_previous_count",
            "device_previous_count",
            "merchant_previous_count"
        ]
    ].to_string(index=False)
)

print("\nPrevious transactions:", len(previous))