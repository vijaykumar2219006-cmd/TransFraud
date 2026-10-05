import numpy as np
import pandas as pd

# ------------------------------------------
# Load test sequences
# ------------------------------------------

data = np.load(
    "datasets/SyntheticBanking/sequences/test_sequences.npz"
)

X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# Find first fraud sequence
fraud_positions = np.where(y == 1)[0]

print("Number of fraud sequences:", len(fraud_positions))

position = fraud_positions[0]

print("\nFirst fraud sequence position:", position)

print("Sequence:")
print(X[position])

# ------------------------------------------
# Load test CSV
# ------------------------------------------

test = pd.read_csv(
    "datasets/SyntheticBanking/test.csv"
)

print("\nTest CSV shape:", test.shape)

print("\nFirst fraud transactions in test.csv:")

fraud_test = test[
    test["is_fraud"] == 1
]

print(
    fraud_test[
        [
            "customer_id",
            "amount",
            "is_fraud",
            "fraud_type"
        ]
    ].head()
)

# ------------------------------------------
# Labels around sequence
# ------------------------------------------

print("\nFirst 20 test labels:")

print(y[:20])

print("\nFraud positions:")

print(fraud_positions[:20])