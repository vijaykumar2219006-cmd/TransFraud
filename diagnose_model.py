import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "transfraud_bilstm_transformer.keras"
TEST_PATH = "datasets/SyntheticBanking/sequences/test_sequences.npz"

model = load_model(MODEL_PATH)

data = np.load(TEST_PATH)
X = data["X"]
y = data["y"]

feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

feature_std[feature_std == 0] = 1

index = 286

sequence = np.squeeze(X[index].copy())

print("Sequence shape:", sequence.shape)
print("Actual label:", y[index])

# Use the customer's average historical amount
customer_avg_amount = 51.54

print("\nCORRECTED AMOUNT TEST")
print("=====================")

for amount in [100, 500, 5000, 10000, 50000]:

    test_sequence = sequence.copy()

    # Current transaction amount
    test_sequence[-1, 0] = amount

    # IMPORTANT:
    # Recalculate the dependent engineered feature
    test_sequence[-1, 3] = abs(
        amount - customer_avg_amount
    )

    # Normal device
    test_sequence[-1, 6] = 0

    scaled = (
        test_sequence - feature_mean
    ) / feature_std

    scaled = scaled.reshape(1, 10, 12)

    probability = float(
        model.predict(
            scaled,
            verbose=0
        )[0][0]
    )

    print(
        f"Amount: {amount:>6} | "
        f"Deviation: {test_sequence[-1,3]:>10.2f} | "
        f"Probability: {probability:.4f}"
    )


print("\nCORRECTED NEW DEVICE TEST")
print("=========================")

amount = 5000

for new_device in [0, 1]:

    test_sequence = sequence.copy()

    test_sequence[-1, 0] = amount

    test_sequence[-1, 3] = abs(
        amount - customer_avg_amount
    )

    test_sequence[-1, 6] = new_device

    scaled = (
        test_sequence - feature_mean
    ) / feature_std

    scaled = scaled.reshape(1, 10, 12)

    probability = float(
        model.predict(
            scaled,
            verbose=0
        )[0][0]
    )

    print(
        f"Amount: {amount} | "
        f"New Device: {new_device} | "
        f"Probability: {probability:.4f}"
    )