import numpy as np
from tensorflow.keras.models import load_model

# Load validation data
val = np.load(
    "datasets/SyntheticBanking/sequences/validation_sequences.npz"
)

X_val = val["X"].astype("float32")

# Load scaling values
mean = np.load("feature_mean.npy")
std = np.load("feature_std.npy")
std[std == 0] = 1

X_val = (X_val - mean) / std

# Load trained model
model = load_model(
    "transfraud_bilstm_transformer.keras"
)

# Predict fraud probability
y_prob = model.predict(
    X_val,
    batch_size=256
).ravel()

# Risk thresholds
T1 = 0.50
T2 = 0.95

# Make payment decisions
decisions = np.where(
    y_prob < T1,
    "Proceed",
    np.where(
        y_prob < T2,
        "Verify",
        "Alert / Verify"
    )
)

# Count decisions
proceed = np.sum(decisions == "Proceed")
verify = np.sum(decisions == "Verify")
alert = np.sum(decisions == "Alert / Verify")

print("\n========== PAYMENT DECISION ==========")
print(f"Proceed        : {proceed}")
print(f"Verify         : {verify}")
print(f"Alert / Verify : {alert}")

# Show examples
print("\nSample decisions:")

for i in range(10):
    print(
        f"Transaction {i+1}: "
        f"Probability = {y_prob[i]:.4f}, "
        f"Decision = {decisions[i]}"
    )