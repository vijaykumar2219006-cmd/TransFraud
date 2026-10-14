
import pandas as pd

df = pd.read_csv(
    "datasets/SyntheticBanking/Fraud Detection Dataset.csv"
)

customer_id = "CUST0000069"

customer_data = df[
    df["customer_id"].astype(str) == customer_id
]

print("Customer exists:", not customer_data.empty)

if not customer_data.empty:
    print("\nMerchants used by this customer:")

    result = (
        customer_data.groupby("merchant_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            latest_transaction=("timestamp", "max")
        )
        .sort_values("transaction_count", ascending=False)
    )

    print(result)

print("\nSample valid merchant IDs:")
print(df["merchant_id"].dropna().unique()[:10])
