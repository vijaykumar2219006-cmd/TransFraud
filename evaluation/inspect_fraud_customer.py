import pandas as pd

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["customer_id", "timestamp"])

# Fraud transactions that have previous customer transactions
fraud_with_history = df[
    (df["is_fraud"] == 1) &
    (df["customer_previous_count"] > 0)
]

print(
    "Fraud transactions with previous history:",
    len(fraud_with_history)
)

print()

print(
    fraud_with_history[
        [
            "customer_id",
            "timestamp",
            "amount",
            "fraud_type",
            "customer_previous_count",
            "card_previous_count",
            "device_previous_count",
            "merchant_previous_count"
        ]
    ].head(10).to_string(index=False)
)