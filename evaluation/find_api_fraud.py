import pandas as pd

df = pd.read_csv("datasets/SyntheticBanking/fraud_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)

fraud = df[df["is_fraud"] == 1]

print("Total fraud transactions:", len(fraud))

print("\nFirst 10 fraud indices:")
print(
    fraud[
        ["customer_id", "amount", "is_fraud", "fraud_type"]
    ].head(10)
)

print("\nActual dataframe indices:")
print(fraud.index[:10].tolist())