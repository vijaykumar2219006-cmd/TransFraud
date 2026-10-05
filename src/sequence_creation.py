import pandas as pd
import numpy as np
import os

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

def create_sequences(input_file, output_file):
    print(f"\nLoading {input_file}...")

    df = pd.read_csv(input_file)

    # Make sure transactions are in chronological order
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    sequences = []
    labels = []

    print("Creating sequences...")

    for customer_id, group in df.groupby("customer_id", sort=False):

        values = group[FEATURES].values.astype(np.float32)
        targets = group["is_fraud"].values

        for i in range(len(group)):

            start = max(0, i - SEQUENCE_LENGTH + 1)

            sequence = values[start:i + 1]

            # Pad beginning if fewer than 10 transactions exist
            if len(sequence) < SEQUENCE_LENGTH:
                padding = np.zeros(
                    (SEQUENCE_LENGTH - len(sequence), len(FEATURES)),
                    dtype=np.float32
                )

                sequence = np.vstack([padding, sequence])

            sequences.append(sequence)
            labels.append(targets[i])

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print("Sequences shape:", X.shape)
    print("Labels shape:", y.shape)

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    print(f"Saved: {output_file}")


os.makedirs("datasets/SyntheticBanking/sequences", exist_ok=True)

create_sequences(
    "datasets/SyntheticBanking/train.csv",
    "datasets/SyntheticBanking/sequences/train_sequences.npz"
)

create_sequences(
    "datasets/SyntheticBanking/validation.csv",
    "datasets/SyntheticBanking/sequences/validation_sequences.npz"
)

create_sequences(
    "datasets/SyntheticBanking/test.csv",
    "datasets/SyntheticBanking/sequences/test_sequences.npz"
)

print("\n========== COMPLETE ==========")