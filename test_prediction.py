import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ==========================================
# 1. Load test data
# ==========================================

df = pd.read_csv(
    "datasets/SyntheticBanking/test.csv"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)

# ==========================================
# 2. Features used by the model
# ==========================================

features = [
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

# ==========================================
# 3. Load trained model
# ==========================================

model = load_model(
    "transfraud_bilstm_transformer.keras"
)

# Load scaling values
mean = np.load("feature_mean.npy")
std = np.load("feature_std.npy")

std[std == 0] = 1

# ==========================================
# 4. Create sequence for a transaction
# ==========================================

def create_sequence(index):

    customer_id = df.loc[index, "customer_id"]

    customer_data = df[
        df["customer_id"] == customer_id
    ]

    current_position = customer_data.index.get_loc(index)

    start = max(0, current_position - 9)

    sequence_indices = customer_data.index[
        start:current_position + 1
    ]

    sequence = df.loc[
        sequence_indices, features
    ].values.astype("float32")

    # Scale
    sequence = (sequence - mean.reshape(1, -1)) / std.reshape(1, -1)

    # Left padding
    if len(sequence) < 10:

        padding = np.zeros(
            (10 - len(sequence), len(features)),
            dtype="float32"
        )

        sequence = np.vstack(
            [padding, sequence]
        )

    return sequence


# ==========================================
# 5. Prediction function
# ==========================================

def predict_transaction(index):

    sequence = create_sequence(index)

    X = np.expand_dims(sequence, axis=0)

    probability = float(
        model.predict(X, verbose=0)[0][0]
    )

    # Risk classification
    if probability < 0.50:
        risk = "Low"
        decision = "Proceed"

    elif probability < 0.95:
        risk = "Medium"
        decision = "Verify"

    else:
        risk = "High"
        decision = "Alert / Verify"

    actual = df.loc[index, "is_fraud"]

    actual_label = (
        "Fraud" if actual == 1
        else "Legitimate"
    )

    print("\n-----------------------------------")
    print("Transaction Index :", index)
    print("Actual         :", actual_label)
    print(f"Fraud Probability : {probability:.4f}")
    print("Risk           :", risk)
    print("Decision       :", decision)


# ==========================================
# 6. Select transactions
# ==========================================

fraud_indices = df.index[
    df["is_fraud"] == 1
]

legitimate_indices = df.index[
    df["is_fraud"] == 0
]

# Pick one fraud and one legitimate transaction
fraud_index = fraud_indices[0]
legitimate_index = legitimate_indices[0]

print("\n========== FRAUD TRANSACTION ==========")
predict_transaction(fraud_index)

print("\n========== LEGITIMATE TRANSACTION ==========")
predict_transaction(legitimate_index)