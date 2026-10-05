import pandas as pd

file_path = r"datasets\SyntheticBanking\fraud_features.csv"

print("Loading feature-engineered dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded.")
print("Total transactions:", len(df))


# ==========================================
# 1. SPLIT FRAUD AND LEGITIMATE
# ==========================================

fraud = df[df["is_fraud"] == 1]
legitimate = df[df["is_fraud"] == 0]

print("\n========== TRANSACTION COUNTS ==========")

print("Fraud:", len(fraud))
print("Legitimate:", len(legitimate))


# ==========================================
# 2. FEATURES TO VALIDATE
# ==========================================

features = [
    "customer_previous_count",
    "customer_avg_previous_amount",
    "customer_amount_deviation",
    "card_previous_count",
    "device_previous_count",
    "new_device",
    "merchant_previous_count",
    "time_since_previous_customer_transaction"
]


# ==========================================
# 3. FRAUD VS LEGITIMATE
# ==========================================

print("\n========== FEATURE COMPARISON ==========")

for feature in features:

    print("\n---", feature, "---")

    print(
        "Fraud:"
    )

    print(
        fraud[feature].describe()
    )

    print(
        "Legitimate:"
    )

    print(
        legitimate[feature].describe()
    )


# ==========================================
# 4. NEW DEVICE ANALYSIS
# ==========================================

print("\n========== NEW DEVICE ANALYSIS ==========")

fraud_new_device = (
    fraud["new_device"].sum()
)

legitimate_new_device = (
    legitimate["new_device"].sum()
)

print(
    "Fraud transactions with new device:",
    fraud_new_device
)

print(
    "Legitimate transactions with new device:",
    legitimate_new_device
)

print(
    "Fraud new-device percentage:",
    (fraud_new_device / len(fraud)) * 100
)

print(
    "Legitimate new-device percentage:",
    (legitimate_new_device / len(legitimate)) * 100
)


# ==========================================
# 5. LARGE AMOUNT DEVIATION
# ==========================================

print("\n========== AMOUNT DEVIATION ==========")

print(
    "Fraud transactions with amount > 5x previous average:",
    (fraud["customer_amount_deviation"] > 5).sum()
)

print(
    "Legitimate transactions with amount > 5x previous average:",
    (legitimate["customer_amount_deviation"] > 5).sum()
)

print(
    "Fraud transactions with amount > 10x previous average:",
    (fraud["customer_amount_deviation"] > 10).sum()
)

print(
    "Legitimate transactions with amount > 10x previous average:",
    (legitimate["customer_amount_deviation"] > 10).sum()
)


# ==========================================
# 6. PREVIOUS CUSTOMER HISTORY
# ==========================================

print("\n========== CUSTOMER HISTORY ==========")

print(
    "Fraud with previous customer transaction:",
    (fraud["customer_previous_count"] > 0).sum()
)

print(
    "Fraud with 5+ previous transactions:",
    (fraud["customer_previous_count"] >= 5).sum()
)

print(
    "Legitimate with previous transaction:",
    (legitimate["customer_previous_count"] > 0).sum()
)

print(
    "Legitimate with 5+ previous transactions:",
    (legitimate["customer_previous_count"] >= 5).sum()
)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n========== VALIDATION COMPLETE ==========")

print(
    "These statistics help us determine which engineered features "
    "may be useful for fraud prediction."
)