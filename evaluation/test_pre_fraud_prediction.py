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

customer_id = "CUST0000069"

# Load data
df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    ["customer_id", "timestamp"]
)

# Get customer's transactions
customer_history = df[
    df["customer_id"] == customer_id
].copy()

customer_history = customer_history.sort_values(
    "timestamp"
)

# Locate fraud transaction
fraud_row = customer_history[
    customer_history["is_fraud"] == 1
].iloc[0]

fraud_time = fraud_row["timestamp"]

# ONLY transactions before fraud
previous = customer_history[
    customer_history["timestamp"] < fraud_time
].copy()

print("Customer:", customer_id)
print("Fraud amount:", fraud_row["amount"])
print("Fraud type:", fraud_row["fraud_type"])
print("Previous transactions:", len(previous))

# Take last 10 previous transactions
sequence_data = previous[FEATURES].tail(
    SEQUENCE_LENGTH
).values.astype(np.float32)

# Padding if necessary
if len(sequence_data) < SEQUENCE_LENGTH:

    padding = np.zeros(
        (
            SEQUENCE_LENGTH - len(sequence_data),
            len(FEATURES)
        ),
        dtype=np.float32
    )

    sequence_data = np.vstack([
        padding,
        sequence_data
    ])

# Load scaling
feature_mean = np.load(MEAN_PATH)
feature_std = np.load(STD_PATH)

# Scale
sequence_data = (
    sequence_data - feature_mean.reshape(1, 1, -1)
) / feature_std.reshape(1, 1, -1)

# Ensure shape is exactly (1, 10, 12)
if sequence_data.ndim == 2:
    sequence_data = np.expand_dims(
        sequence_data,
        axis=0
    )

print("Final sequence shape:", sequence_data.shape)

# Load model
model = load_model(MODEL_PATH)

# Predict
prediction = model.predict(
    sequence_data,
    verbose=0
)

probability = float(
    prediction[0][0]
)

print("\nPre-Fraud Prediction")
print("--------------------")
print("Actual fraud:", 1)
print(
    "Fraud probability:",
    round(probability, 4)
)