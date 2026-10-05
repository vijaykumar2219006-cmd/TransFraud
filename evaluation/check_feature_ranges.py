import pandas as pd

PATH = "datasets/SyntheticBanking/fraud_features.csv"

df = pd.read_csv(PATH)

features = [
    "amount",
    "customer_previous_count",
    "customer_avg_previous_amount",
    "customer_amount_deviation",
    "card_previous_count",
    "device_previous_count",
    "new_device",
    "merchant_previous_count",
    "time_since_previous_customer_transaction"
]

print("\nTRAINING DATA FEATURE RANGES")
print("============================")

for feature in features:

    print(f"\n{feature}")

    print("Min:", df[feature].min())
    print("Max:", df[feature].max())
    print("Mean:", df[feature].mean())
    print("Median:", df[feature].median())

    print(
        "95th percentile:",
        df[feature].quantile(0.95)
    )

    print(
        "99th percentile:",
        df[feature].quantile(0.99)
    )