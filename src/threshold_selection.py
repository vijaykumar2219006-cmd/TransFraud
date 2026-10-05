import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# Load validation data
val = np.load(
    "datasets/SyntheticBanking/sequences/validation_sequences.npz"
)

X_val = val["X"].astype("float32")
y_val = val["y"]

# Load training scaling values
mean = np.load("feature_mean.npy")
std = np.load("feature_std.npy")
std[std == 0] = 1

X_val = (X_val - mean) / std

# Load trained model
model = load_model(
    "transfraud_bilstm_transformer.keras"
)

# Get fraud probabilities
y_prob = model.predict(X_val, batch_size=256).ravel()

# Test different thresholds
thresholds = np.arange(0.05, 1.00, 0.05)

print("\n========== THRESHOLD ANALYSIS ==========")
print("Threshold | Precision | Recall | F1")
print("----------------------------------------")

best_threshold = 0
best_f1 = 0

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    print(
        f"{threshold:.2f}      | "
        f"{precision:.4f}    | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n========== RESULT ==========")
print(f"Best threshold : {best_threshold:.2f}")
print(f"Best F1 score  : {best_f1:.4f}")