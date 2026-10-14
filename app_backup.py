import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "models/transfraud_bilstm_transformer.keras"

DATA_PATH = "datasets/SyntheticBanking/fraud_features.csv"

MEAN_PATH = "models/feature_mean.npy"
STD_PATH = "models/feature_std.npy"

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
    card_id: str
    merchant_id: str
    device_id: str
    amount: float
    


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

def create_new_transaction_features(
    customer_id,
    card_id,
    merchant_id,
    device_id,
    amount
):
    print("DEBUG: create_new_transaction_features called", flush=True)

    # ==========================================
    # CUSTOMER HISTORY
    # ==========================================

    customer_history = df[
        df["customer_id"] == customer_id
    ].copy()

    customer_history = customer_history.sort_values(
        "timestamp"
    )

    # ==========================================
    # CUSTOMER FEATURES
    # ==========================================

    if len(customer_history) == 0:
        # Completely new customer
        customer_previous_count = 0
        customer_avg_previous_amount = 0
        customer_amount_deviation = 0

        # No previous transaction exists
        previous_timestamp = None

    else:
        # Existing customer
        customer_previous_count = len(customer_history)

        customer_avg_previous_amount = (
            customer_history["amount"].mean()
        )

        if customer_avg_previous_amount > 0:
            customer_amount_deviation = (
                amount / customer_avg_previous_amount
            )
        else:
            customer_amount_deviation = 0

        previous_timestamp = customer_history.iloc[-1]["timestamp"]

    # ==========================================
    # HISTORICAL CARD / DEVICE / MERCHANT DATA
    # ==========================================

    if previous_timestamp is None:

        card_previous_count = 0
        device_previous_count = 0
        merchant_previous_count = 0

    else:

        card_history = df[
            (df["card_id"].astype(str) == str(card_id)) &
            (df["timestamp"] <= previous_timestamp)
        ]

        card_previous_count = len(card_history)

        device_history = df[
            (df["device_id"].astype(str) == str(device_id)) &
            (df["timestamp"] <= previous_timestamp)
        ]

        device_previous_count = len(device_history)

        merchant_history = df[
            (df["merchant_id"].astype(str) == str(merchant_id)) &
            (df["timestamp"] <= previous_timestamp)
        ]

        merchant_previous_count = len(merchant_history)


    # ==========================================
    # ESTIMATE TRANSACTION TIME
    # ==========================================

    if previous_timestamp is None:
        # Completely new customer
        current_timestamp = pd.Timestamp.now(tz="UTC")
        time_since_previous = 0

    else:
        # Existing customer
        if len(customer_history) > 1:

            customer_gaps = (
                customer_history["timestamp"]
                .diff()
                .dt.total_seconds()
                .dropna()
            )

            customer_gaps = customer_gaps[
                customer_gaps > 0
            ]

            if len(customer_gaps) > 0:
                estimated_gap = customer_gaps.median()
            else:
                estimated_gap = 86400

        else:
            estimated_gap = 86400

        current_timestamp = (
            previous_timestamp +
            pd.Timedelta(seconds=float(estimated_gap))
        )

        time_since_previous = estimated_gap

        # TEMPORARY DEBUG: compare device-history cutoffs
    device_rows = df[
        df["device_id"].astype(str) == str(device_id)
    ]

    count_at_previous = len(
        device_rows[
            device_rows["timestamp"] <= previous_timestamp
        ]
    ) if previous_timestamp is not None else 0

    count_at_estimated_current = len(
        device_rows[
            device_rows["timestamp"] < current_timestamp
        ]
    )

    print("DEBUG device history at previous timestamp:",
          count_at_previous, flush=True)

    print("DEBUG device history before estimated current time:",
          count_at_estimated_current, flush=True)

    # ==========================================
    # TEMPORAL FEATURES
    # ==========================================

    hour = current_timestamp.hour

    day_of_week = current_timestamp.dayofweek

    month = current_timestamp.month

    # ==========================================
    # FINAL 12 FEATURES
    # ==========================================
    print("Customer history:", customer_previous_count)
    print("Card history:", card_previous_count)
    print("Device history:", device_previous_count)
    print("New device:", int(device_previous_count == 0))
    print("Merchant history:", merchant_previous_count)
    print("Amount:", amount)

    features = np.array([
        amount,
        customer_previous_count,
        customer_avg_previous_amount,
        customer_amount_deviation,
        card_previous_count,
        device_previous_count,
        int(device_previous_count == 0),
        merchant_previous_count,
        time_since_previous,
        hour,
        day_of_week,
        month
    ], dtype=np.float32)

    return features

class DemoPaymentRequest(BaseModel):
    amount: float
    new_device: int = 0
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

# ==================================================
# DEMO FRAUD CASE
# ==================================================

@app.get("/demo-fraud-case")
def demo_fraud_case():

    try:

        # Confirmed fraud transaction used for demonstration
        demo_index = 1804

        sequence = create_sequence(demo_index)

        prediction = model.predict(
            sequence,
            verbose=0
        )

        fraud_probability = float(prediction[0][0])

        risk = classify_risk(fraud_probability)

        decision = payment_decision(risk)

        transaction = df.iloc[demo_index]

        return {
            "demo": True,
            "transaction_index": demo_index,
            "customer_id": str(transaction["customer_id"]),
            "amount": float(transaction["amount"]),
            "actual_label": int(transaction["is_fraud"]),
            "fraud_probability": round(fraud_probability, 4),
            "risk": risk,
            "decision": decision
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
    payment.card_id,
    payment.merchant_id,
    payment.device_id,
    payment.amount
)

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

        print("DEBUG RAW SEQUENCE:", sequence.tolist(), flush=True)

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

        # ------------------------------------------
        # Add batch dimension
        # ------------------------------------------

        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)


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
        import traceback
        traceback.print_exc()
        return {
            "error": str(e)
        }
@app.post("/predict-demo-payment")
def predict_demo_payment(payment: DemoPaymentRequest):

    try:

        # ------------------------------------------
        # Demo customer
        # ------------------------------------------

        customer_id = "CUST0000069"

        customer_history = df[
            df["customer_id"] == customer_id
        ].copy()

        customer_history = customer_history.sort_values(
            "timestamp"
        )

        # ------------------------------------------
        # Find the known fraud transaction
        # ------------------------------------------

        fraud_rows = customer_history[
            customer_history["is_fraud"] == 1
        ]

        if len(fraud_rows) == 0:
            raise ValueError(
                "No fraud transaction found for demo customer."
            )

        fraud_transaction = fraud_rows.iloc[0]

        fraud_timestamp = fraud_transaction["timestamp"]

        # ------------------------------------------
        # ONLY transactions before the fraud
        # ------------------------------------------

        pre_fraud_history = customer_history[
            customer_history["timestamp"] < fraud_timestamp
        ].copy()

        if len(pre_fraud_history) == 0:
            raise ValueError(
                "No historical transactions before fraud."
            )

        # ------------------------------------------
        # Last 9 historical transactions
        # ------------------------------------------

        history = pre_fraud_history[
            FEATURES
        ].tail(SEQUENCE_LENGTH - 1).values.astype(
            np.float32
        )

        # ------------------------------------------
        # Current transaction
        #
        # Use the actual transaction's model features.
        # DO NOT use is_fraud or fraud_type.
        # ------------------------------------------

        current_features = fraud_transaction[
            FEATURES
        ].values.astype(np.float32)

        # ------------------------------------------
        # Allow demo amount/device modification
        # ------------------------------------------

        current_features[0] = payment.amount

        # Recalculate amount deviation
        previous_avg = (
            pre_fraud_history["amount"].mean()
        )

        current_features[3] = abs(
            payment.amount - previous_avg
        )

        # ------------------------------------------
        # Create sequence
        # ------------------------------------------

        sequence = np.vstack([
            history,
            current_features
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

        # ------------------------------------------
        # Batch dimension
        # ------------------------------------------

        sequence = sequence.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURES)
        )

        print("\nDemo sequence shape:")
        print(sequence.shape)

        print("\nDemo current transaction:")
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

        # ------------------------------------------
        # Risk classification
        # ------------------------------------------

        risk = classify_risk(
            fraud_probability
        )

        decision = payment_decision(
            risk
        )

        # ------------------------------------------
        # Response
        # ------------------------------------------

        return {
            "customer_id": customer_id,
            "amount": payment.amount,
            "fraud_probability":
                round(fraud_probability, 4),
            "risk": risk,
            "decision": decision,
            "checked_before_payment": True
        }

    except Exception as e:

        return {
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)