import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = r"datasets\SyntheticBanking\Fraud Detection Dataset.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded.")
print("Rows:", len(df))


# ==========================================
# 2. CONVERT TIMESTAMP
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# ==========================================
# 3. SORT CHRONOLOGICALLY
# ==========================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Data sorted by timestamp.")


# ==========================================
# 4. CUSTOMER HISTORY FEATURES
# ==========================================

print("Creating customer features...")

df["customer_previous_count"] = (
    df.groupby("customer_id").cumcount()
)

df["customer_previous_amount_sum"] = (
    df.groupby("customer_id")["amount"]
    .cumsum()
    - df["amount"]
)

df["customer_avg_previous_amount"] = np.where(
    df["customer_previous_count"] > 0,
    df["customer_previous_amount_sum"] /
    df["customer_previous_count"],
    0
)


# ==========================================
# 5. CARD HISTORY
# ==========================================

print("Creating card features...")

df["card_previous_count"] = (
    df.groupby("card_id").cumcount()
)


# ==========================================
# 6. DEVICE HISTORY
# ==========================================

print("Creating device features...")

df["device_previous_count"] = (
    df.groupby("device_id").cumcount()
)


# New device indicator
df["new_device"] = (
    df["device_previous_count"] == 0
).astype(int)


# ==========================================
# 7. MERCHANT HISTORY
# ==========================================

print("Creating merchant features...")

df["merchant_previous_count"] = (
    df.groupby("merchant_id").cumcount()
)


# ==========================================
# 8. TIME SINCE PREVIOUS CUSTOMER TRANSACTION
# ==========================================

print("Creating time-based features...")

df["previous_customer_timestamp"] = (
    df.groupby("customer_id")["timestamp"]
    .shift(1)
)

df["time_since_previous_customer_transaction"] = (
    df["timestamp"] -
    df["previous_customer_timestamp"]
).dt.total_seconds()

# First transaction has no previous transaction
df["time_since_previous_customer_transaction"] = (
    df["time_since_previous_customer_transaction"]
    .fillna(-1)
)


# ==========================================
# 9. TEMPORAL FEATURES
# ==========================================

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["month"] = (
    df["timestamp"].dt.month
)


# ==========================================
# 10. CUSTOMER AMOUNT DEVIATION
# ==========================================

print("Creating amount deviation feature...")

df["customer_amount_deviation"] = np.where(
    df["customer_avg_previous_amount"] > 0,
    df["amount"] /
    df["customer_avg_previous_amount"],
    0
)


# ==========================================
# 11. DISPLAY NEW FEATURES
# ==========================================

new_features = [
    "customer_previous_count",
    "customer_avg_previous_amount",
    "card_previous_count",
    "device_previous_count",
    "new_device",
    "merchant_previous_count",
    "time_since_previous_customer_transaction",
    "hour",
    "day_of_week",
    "month",
    "customer_amount_deviation"
]

print("\n========== NEW FEATURES ==========")

print(df[new_features].head(10))


# ==========================================
# 12. CHECK FOR MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")

print(
    df[new_features].isnull().sum()
)


# ==========================================
# 13. SAVE FEATURE-ENGINEERED DATA
# ==========================================

output_file = (
    r"datasets\SyntheticBanking"
    r"\fraud_features.csv"
)

print("\nSaving feature-engineered dataset...")

df.to_csv(
    output_file,
    index=False
)

print("\n========== COMPLETE ==========")

print(
    "Feature-engineered dataset saved to:"
)

print(output_file)