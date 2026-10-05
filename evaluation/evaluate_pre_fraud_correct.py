import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"
MODEL_PATH = "transfraud_bilstm_transformer.keras"
MEAN_PATH = "feature_mean.npy"
STD_PATH = "feature_std.npy"

SEQUENCE_LENGTH = 10

FEATURES = [
    "amount",
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

# -----------------------------------------
# Load data
# -----------------------------------------

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    ["customer_id", "timestamp"]
)

# -----------------------------------------
# Load model and scaler
# -----------------------------------------

model = load_model(MODEL_PATH)

feature_mean = np.load(MEAN_PATH)
feature_std = np.load(STD_PATH)

feature_mean = feature_mean.reshape(1, 1, -1)
feature_std = feature_std.reshape(1, 1, -1)

# -----------------------------------------
# Fraud transactions with history
# -----------------------------------------

fraud_rows = df[
    (df["is_fraud"] == 1) &
    (df["customer_previous_count"] > 0)
].copy()

print(
    "Fraud transactions with history:",
    len(fraud_rows)
)

results = []

# -----------------------------------------
# Evaluate
# -----------------------------------------

for _, fraud_row in fraud_rows.iterrows():

    customer_id = fraud_row["customer_id"]
    fraud_timestamp = fraud_row["timestamp"]

    customer_history = df[
        df["customer_id"] == customer_id
    ].sort_values("timestamp")

    # Previous transactions only
    previous = customer_history[
        customer_history["timestamp"] < fraud_timestamp
    ]

    # Last 9 previous transactions
    previous_sequence = previous[
        FEATURES
    ].tail(SEQUENCE_LENGTH - 1).values.astype(
        np.float32
    )

    # Current fraud transaction
    current_transaction = fraud_row[
        FEATURES
    ].values.astype(np.float32)

    current_transaction = current_transaction.reshape(
        1, -1
    )

    # -----------------------------------------
    # Combine history + current transaction
    # -----------------------------------------

    sequence = np.vstack([
        previous_sequence,
        current_transaction
    ])

    # -----------------------------------------
    # Padding
    # -----------------------------------------

    if len(sequence) < SEQUENCE_LENGTH:

        padding = np.zeros(
            (
                SEQUENCE_LENGTH - len(sequence),
                len(FEATURES)
            ),
            dtype=np.float32
        )

        sequence = np.vstack([
            padding,
            sequence
        ])

    # -----------------------------------------
    # Scale
    # -----------------------------------------

    sequence = sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        len(FEATURES)
    )

    sequence = (
        sequence - feature_mean
    ) / feature_std

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    probability = float(
        model.predict(
            sequence,
            verbose=0
        )[0][0]
    )

    results.append({
        "customer_id": customer_id,
        "fraud_timestamp": fraud_timestamp,
        "amount": fraud_row["amount"],
        "fraud_type": fraud_row["fraud_type"],
        "previous_transactions": len(previous),
        "fraud_probability": probability,
        "actual_label": int(fraud_row["is_fraud"])
    })

# -----------------------------------------
# Results
# -----------------------------------------

results_df = pd.DataFrame(results)

print("\nProbability statistics:")
print(
    results_df["fraud_probability"].describe()
)

# -----------------------------------------
# Threshold evaluation
# -----------------------------------------

for threshold in [0.50, 0.95]:

    detected = (
        results_df["fraud_probability"] >= threshold
    ).sum()

    detection_rate = (
        detected / len(results_df)
    )

    print("\nThreshold:", threshold)
    print("Detected:", detected)
    print(
        "Detection rate:",
        round(detection_rate * 100, 2),
        "%"
    )

# -----------------------------------------
# Top cases
# -----------------------------------------

print("\nTop 10 fraud cases:")

print(
    results_df.sort_values(
        "fraud_probability",
        ascending=False
    ).head(10).to_string(index=False)
)

# -----------------------------------------
# Save
# -----------------------------------------

results_df.to_csv(
    "pre_fraud_correct_results.csv",
    index=False
)

print(
    "\nSaved: pre_fraud_correct_results.csv"
)