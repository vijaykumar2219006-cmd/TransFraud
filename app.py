from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "transfraud_bilstm_transformer.keras"

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"

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


# ==================================================
# LOAD MODEL AND DATA
# ==================================================

print("Loading TransFraud model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("Loading test transactions...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["customer_id", "timestamp"]
).reset_index(drop=True)

print("Transactions loaded:", len(df))


# ==================================================
# LOAD TRAINING SCALING VALUES
# ==================================================

feature_mean = np.load(MEAN_PATH)
feature_std = np.load(STD_PATH)

feature_std[feature_std == 0] = 1


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="TransFraud Payment System",
    description="Pre-payment financial fraud risk detection system"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# INPUT
# ==================================================

class PaymentRequest(BaseModel):

    transaction_index: int

class NewPaymentRequest(BaseModel):
    customer_id: str
    receiver_id: str
    amount: float
    new_device: int = 0


# ==================================================
# RISK CLASSIFICATION
# ==================================================

def classify_risk(probability):

    if probability < 0.50:
        return "Low"

    elif probability < 0.95:
        return "Medium"

    else:
        return "High"


# ==================================================
# PAYMENT DECISION
# ==================================================

def payment_decision(risk):

    if risk == "Low":
        return "Proceed"

    elif risk == "Medium":
        return "Verify"

    else:
        return "Alert / Verify"


# ==================================================
# CREATE REAL CUSTOMER SEQUENCE
# ==================================================

def create_sequence(transaction_index):

    if transaction_index < 0 or transaction_index >= len(df):

        raise ValueError("Invalid transaction index.")

    current_row = df.iloc[transaction_index]

    customer_id = current_row["customer_id"]

    # Get all transactions of this customer
    customer_data = df[
        df["customer_id"] == customer_id
    ].copy()

    customer_data = customer_data.sort_values(
        "timestamp"
    )

    # Find the current transaction
    current_position = customer_data.index.get_loc(
        transaction_index
    )

    start = max(
        0,
        current_position - SEQUENCE_LENGTH + 1
    )

    sequence_data = customer_data.iloc[
        start:current_position + 1
    ]

    values = sequence_data[
        FEATURES
    ].values.astype(np.float32)

    # ----------------------------------------------
    # Padding
    # ----------------------------------------------

    if len(values) < SEQUENCE_LENGTH:

        padding = np.zeros(
            (
                SEQUENCE_LENGTH - len(values),
                len(FEATURES)
            ),
            dtype=np.float32
        )

        values = np.vstack(
            [padding, values]
        )

    # ----------------------------------------------
    # Scale using training statistics
    # ----------------------------------------------

    values = (
        values - feature_mean
    ) / feature_std

    

    return values

def create_new_transaction_features(customer_id, amount, new_device):

    customer_history = df[
        df["customer_id"] == customer_id
    ].copy()

    customer_history = customer_history.sort_values(
        "timestamp"
    )

    if len(customer_history) == 0:
        raise ValueError(
            "Customer not found in transaction history."
        )

    latest = customer_history.iloc[-1]

    # ------------------------------------------
    # Customer behavioral features
    # ------------------------------------------

    customer_previous_count = len(customer_history)

    customer_avg_previous_amount = (
        customer_history["amount"].mean()
    )

    customer_amount_deviation = abs(
        amount - customer_avg_previous_amount
    )

    # ------------------------------------------
    # Historical card information
    # ------------------------------------------

    card_previous_count = float(
        latest["card_previous_count"]
    )

    # ------------------------------------------
    # Device information
    # ------------------------------------------

    if new_device == 1:
        # A new device has no previous transactions
        # associated with it.
        device_previous_count = 0.0
    else:
        device_previous_count = float(
            latest["device_previous_count"]
        )

    # ------------------------------------------
    # Merchant information
    # ------------------------------------------

    merchant_previous_count = float(
        latest["merchant_previous_count"]
    )

    # ------------------------------------------
    # Estimate realistic transaction time
    # ------------------------------------------

    timestamps = (
        customer_history["timestamp"]
        .sort_values()
    )

    if len(timestamps) >= 2:

        time_differences = (
            timestamps.diff()
            .dt.total_seconds()
            .dropna()
        )

        # Remove invalid/non-positive intervals
        time_differences = time_differences[
            time_differences > 0
        ]

        if len(time_differences) > 0:
            estimated_gap = float(
                time_differences.median()
            )
        else:
            estimated_gap = 3600.0

    else:
        # Fallback: 1 hour
        estimated_gap = 3600.0

    previous_timestamp = latest["timestamp"]

    current_timestamp = (
        previous_timestamp
        + pd.Timedelta(seconds=estimated_gap)
    )

    time_since_previous = estimated_gap

    # ------------------------------------------
    # Temporal features
    # ------------------------------------------

    hour = current_timestamp.hour

    day_of_week = current_timestamp.dayofweek

    month = current_timestamp.month

    # ------------------------------------------
    # 12 model features
    # ------------------------------------------

    features = np.array([
        amount,
        customer_previous_count,
        customer_avg_previous_amount,
        customer_amount_deviation,
        card_previous_count,
        device_previous_count,
        new_device,
        merchant_previous_count,
        time_since_previous,
        hour,
        day_of_week,
        month
    ], dtype=np.float32)

    return features

# ==================================================
# PREDICT PAYMENT
# ==================================================

@app.post("/predict-payment")
def predict_payment(payment: PaymentRequest):

    try:

        # Create actual historical sequence
        sequence = create_sequence(
            payment.transaction_index
        )

        # Model prediction
        prediction = model.predict(
            sequence,
            verbose=0
        )

        fraud_probability = float(
            prediction[0][0]
        )

        # Risk
        risk = classify_risk(
            fraud_probability
        )

        # Payment decision
        decision = payment_decision(
            risk
        )

        # Current transaction
        transaction = df.iloc[
            payment.transaction_index
        ]

        return {

            "transaction_index":
                payment.transaction_index,

            "customer_id":
                str(transaction["customer_id"]),

            "amount":
                float(transaction["amount"]),

            "actual_label":
                int(transaction["is_fraud"]),

            "fraud_probability":
                round(
                    fraud_probability,
                    4
                ),

            "risk":
                risk,

            "decision":
                decision
        }

    except Exception as e:

        return {
            "error": str(e)
        }
@app.post("/predict-new-payment")
def predict_new_payment(payment: NewPaymentRequest):

    try:

        # Create features for new transaction
        new_features = create_new_transaction_features(
            payment.customer_id,
            payment.amount,
            payment.new_device
        )

        print("\nNew transaction features:")
        print(new_features)

        # ------------------------------------------
        # Get customer's previous transactions
        # ------------------------------------------

        customer_history = df[
            df["customer_id"] == payment.customer_id
        ].copy()

        customer_history = customer_history.sort_values(
            "timestamp"
        )

        # Use the last 9 historical transactions
        history = customer_history[
            FEATURES
        ].tail(SEQUENCE_LENGTH - 1).values.astype(
            np.float32
        )

        # ------------------------------------------
        # Add new transaction
        # ------------------------------------------

        sequence = np.vstack([
            history,
            new_features
        ])

        # ------------------------------------------
        # Padding
        # ------------------------------------------

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

        # ------------------------------------------
        # Scale
        # ------------------------------------------

        sequence = (
            sequence - feature_mean
        ) / feature_std
        print("\nScaled new transaction:")
        print(sequence[0, -1])

        # ------------------------------------------
        # Add batch dimension
        # ------------------------------------------

        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        print("Final sequence shape:", sequence.shape)
        print("Final transaction row:")
        print(sequence[0, -1])
        # ------------------------------------------
        # Model prediction
        # ------------------------------------------

        prediction = model.predict(
            sequence,
            verbose=0
        )

        fraud_probability = float(
            prediction[0][0]
        )

        risk = classify_risk(
            fraud_probability
        )

        decision = payment_decision(
            risk
        )

        return {
    "customer_id": payment.customer_id,
    "receiver_id": payment.receiver_id,
    "amount": payment.amount,
    "fraud_probability": round(fraud_probability, 4),
    "risk": risk,
    "decision": decision
}

    except Exception as e:

        return {
            "error": str(e)
        } 