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

# --------------------------------
# Risk thresholds
# --------------------------------

T1 = 0.50
T2 = 0.95

# Risk classification
risk = np.where(
    y_prob < T1,
    "Low",
    np.where(
        y_prob < T2,
        "Medium",
        "High"
    )
)

# Count each risk level
low_count = np.sum(risk == "Low")
medium_count = np.sum(risk == "Medium")
high_count = np.sum(risk == "High")

print("\n========== RISK CLASSIFICATION ==========")
print(f"Low Risk    : {low_count}")
print(f"Medium Risk : {medium_count}")
print(f"High Risk   : {high_count}")

print("\nThresholds:")
print(f"Low < {T1}")
print(f"Medium: {T1} - {T2}")
print(f"High >= {T2}")

# Show sample predictions
print("\nSample predictions:")

for i in range(10):
    print(
        f"Transaction {i+1}: "
        f"Probability = {y_prob[i]:.4f}, "
        f"Risk = {risk[i]}"
    )