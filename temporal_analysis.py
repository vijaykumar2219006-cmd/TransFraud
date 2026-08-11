import pandas as pd

file_path = r"datasets\SyntheticBanking\Fraud Detection Dataset.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

print("\n========== BASIC TIME INFORMATION ==========")

print("First transaction:", df["timestamp"].min())
print("Last transaction:", df["timestamp"].max())

print("\nTotal transactions:", len(df))

# --------------------------------------------------
# Historical transaction count BEFORE current transaction
# --------------------------------------------------

df["customer_previous_count"] = (
    df.groupby("customer_id").cumcount()
)

df["card_previous_count"] = (
    df.groupby("card_id").cumcount()
)

df["device_previous_count"] = (
    df.groupby("device_id").cumcount()
)

df["merchant_previous_count"] = (
    df.groupby("merchant_id").cumcount()
)

# --------------------------------------------------
# Fraud transactions
# --------------------------------------------------

fraud = df[df["is_fraud"] == 1]

print("\n========== FRAUD HISTORY ANALYSIS ==========")

print("Fraud transactions:", len(fraud))

print("\nCustomer history before fraud:")

print(
    fraud["customer_previous_count"].describe()
)

print("\nCard history before fraud:")

print(
    fraud["card_previous_count"].describe()
)

print("\nDevice history before fraud:")

print(
    fraud["device_previous_count"].describe()
)

print("\nMerchant history before fraud:")

print(
    fraud["merchant_previous_count"].describe()
)

# --------------------------------------------------
# How many fraud transactions have historical data?
# --------------------------------------------------

print("\n========== AVAILABLE HISTORY ==========")

print(
    "Fraud transactions with previous customer transaction:",
    (fraud["customer_previous_count"] > 0).sum()
)

print(
    "Fraud transactions with 5+ previous customer transactions:",
    (fraud["customer_previous_count"] >= 5).sum()
)

print(
    "Fraud transactions with 10+ previous customer transactions:",
    (fraud["customer_previous_count"] >= 10).sum()
)

print(
    "Fraud transactions with previous card transaction:",
    (fraud["card_previous_count"] > 0).sum()
)

print(
    "Fraud transactions with previous device transaction:",
    (fraud["device_previous_count"] > 0).sum()
)

print(
    "Fraud transactions with previous merchant transaction:",
    (fraud["merchant_previous_count"] > 0).sum()
)

# --------------------------------------------------
# Fraud types
# --------------------------------------------------

print("\n========== FRAUD TYPES ==========")

print(
    fraud["fraud_type"].value_counts()
)

# --------------------------------------------------
# Fraud over time
# --------------------------------------------------

print("\n========== FRAUD BY MONTH ==========")

fraud_by_month = (
    fraud
    .set_index("timestamp")
    .resample("ME")
    .size()
)

print(fraud_by_month)

# --------------------------------------------------
# Fraud by hour of day
# --------------------------------------------------

print("\n========== FRAUD BY HOUR ==========")

fraud_by_hour = (
    fraud["timestamp"]
    .dt.hour
    .value_counts()
    .sort_index()
)

print(fraud_by_hour)

# --------------------------------------------------
# Fraud by country
# --------------------------------------------------

print("\n========== FRAUD BY MERCHANT COUNTRY ==========")

print(
    fraud["merchant_country"]
    .value_counts()
    .head(20)
)