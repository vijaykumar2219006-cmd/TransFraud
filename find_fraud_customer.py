import pandas as pd

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"

df = pd.read_csv(DATA_PATH)

fraud_customers = df[
    df["is_fraud"] == 1
]["customer_id"].unique()

print("Fraud customers found:", len(fraud_customers))

for customer in fraud_customers[:10]:
    history = df[
        df["customer_id"] == customer
    ].sort_values("timestamp")

    print(
        customer,
        "transactions:",
        len(history),
        "fraud transactions:",
        history["is_fraud"].sum()
    )