import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

MODEL_PATH = "transfraud_bilstm_transformer.keras"
TEST_PATH = "datasets/SyntheticBanking/sequences/test_sequences.npz"

# Load model
model = load_model(MODEL_PATH)

# Load test sequences
data = np.load(TEST_PATH)

X_test = data["X"]
y_test = data["y"]

print("Test sequence shape:", X_test.shape)
print("Test labels:", len(y_test))

# Load training scaler
feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

# Scale
X_test = (
    X_test - feature_mean
) / feature_std

# Prediction
y_prob = model.predict(
    X_test,
    verbose=0
).ravel()

# ROC-AUC
auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC-AUC:", round(auc, 4))

# Threshold evaluation
for threshold in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n==============================")
    print("Threshold:", threshold)
    print("==============================")

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))

    print("Confusion Matrix:")
    print(cm)