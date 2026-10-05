import pandas as pd

# ==========================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ==========================================

file_path = r"datasets\SyntheticBanking\fraud_features.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Total transactions:", len(df))


# ==========================================
# 2. SORT CHRONOLOGICALLY
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print("Data sorted chronologically.")


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "amount",
    "merchant_category",
    "merchant_country",
    "merchant_city",
    "merchant_latitude",
    "merchant_longitude",
    "transaction_type",
    "customer_previous_count",
    "customer_avg_previous_amount",
    "customer_amount_deviation",
    "card_previous_count",
    "device_previous_count",
    "new_device",
    "merchant_previous_count",
    "time_since_previous_customer_transaction",
    "hour",
    "day_of_week",
    "month"
]

target = "is_fraud"

data = df[["customer_id", "timestamp"] + features + [target]].copy()


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\nChecking missing values...")

print(data.isnull().sum())

# Numerical columns
numerical_features = [
    "amount",
    "merchant_latitude",
    "merchant_longitude",
    "customer_previous_count",
    "customer_avg_previous_amount",
    "customer_amount_deviation",
    "card_previous_count",
    "device_previous_count",
    "new_device",
    "merchant_previous_count",
    "time_since_previous_customer_transaction",
    "hour",
    "day_of_week",
    "month"
]

# Fill numerical missing values
data[numerical_features] = (
    data[numerical_features].fillna(0)
)


# Categorical columns
categorical_features = [
    "merchant_category",
    "merchant_country",
    "merchant_city",
    "transaction_type"
]

# Fill categorical missing values
data[categorical_features] = (
    data[categorical_features].fillna("UNKNOWN")
)


# ==========================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST
# ==========================================

total = len(data)

train_end = int(total * 0.70)
validation_end = int(total * 0.85)

train_data = data.iloc[:train_end].copy()

validation_data = data.iloc[
    train_end:validation_end
].copy()

test_data = data.iloc[
    validation_end:
].copy()


# ==========================================
# 6. DISPLAY SPLIT INFORMATION
# ==========================================

print("\n========== DATA SPLIT ==========")

print(
    "Training transactions:",
    len(train_data)
)

print(
    "Validation transactions:",
    len(validation_data)
)

print(
    "Test transactions:",
    len(test_data)
)

print("\n========== FRAUD DISTRIBUTION ==========")

print(
    "Training fraud:",
    train_data["is_fraud"].sum()
)

print(
    "Validation fraud:",
    validation_data["is_fraud"].sum()
)

print(
    "Test fraud:",
    test_data["is_fraud"].sum()
)


# ==========================================
# 7. SAVE SPLITS
# ==========================================

print("\nSaving datasets...")

train_data.to_csv(
    r"datasets\SyntheticBanking\train.csv",
    index=False
)

validation_data.to_csv(
    r"datasets\SyntheticBanking\validation.csv",
    index=False
)

test_data.to_csv(
    r"datasets\SyntheticBanking\test.csv",
    index=False
)

print("\n========== COMPLETE ==========")

print("Saved:")
print("train.csv")
print("validation.csv")
print("test.csv")