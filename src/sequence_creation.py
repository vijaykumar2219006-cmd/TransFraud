
import os
from collections import deque

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 10

BASE_DIR = "datasets/SyntheticBanking"
OUTPUT_DIR = os.path.join(BASE_DIR, "sequences_corrected")

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
    "month",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep each customer's earlier transactions across split boundaries.
history = {}

for split in ["train", "validation", "test"]:
    input_file = os.path.join(BASE_DIR, f"{split}.csv")
    output_file = os.path.join(
        OUTPUT_DIR, f"{split}_sequences.npz"
    )

    print(f"\nLoading {input_file}...", flush=True)

    df = pd.read_csv(input_file)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    )
    df = df.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    values = df[FEATURES].to_numpy(dtype=np.float32)
    customers = df["customer_id"].to_numpy()
    labels = df["is_fraud"].to_numpy(dtype=np.int64)

    X = np.zeros(
        (len(df), SEQUENCE_LENGTH, len(FEATURES)),
        dtype=np.float32,
    )

    print(f"Creating {split} sequences...", flush=True)

    for i, (customer, current) in enumerate(
        zip(customers, values)
    ):
        if customer not in history:
            history[customer] = deque(
                maxlen=SEQUENCE_LENGTH
            )

        customer_history = history[customer]
        customer_history.append(current)

        seq = np.asarray(
            customer_history, dtype=np.float32
        )

        X[i, -len(seq):, :] = seq

    np.savez_compressed(
        output_file,
        X=X,
        y=labels,
    )

    print("Saved:", output_file, flush=True)
    print("Sequence shape:", X.shape, flush=True)
    print("Fraud labels:", int(labels.sum()), flush=True)

    # Keep history between train -> validation -> test.
    # This allows later splits to use past transactions only.

print("\nCorrected sequences created successfully.")
print("Original sequence files and model were not overwritten.")
