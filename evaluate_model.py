import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load test data
test = np.load(
    "datasets/SyntheticBanking/sequences/test_sequences.npz"
)

X_test = test["X"].astype("float32")
y_test = test["y"]

# Load training statistics
mean = np.load("feature_mean.npy")
std = np.load("feature_std.npy")

std[std == 0] = 1

# Scale test data using training statistics
X_test = (X_test - mean) / std

print("Test data:", X_test.shape)

# Load trained model
model = load_model(
    "transfraud_bilstm_transformer.keras"
)

# Predict fraud probabilities
y_prob = model.predict(X_test, batch_size=256).ravel()

# Default threshold
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n========== TEST RESULTS ==========")
print(f"AUC       : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)