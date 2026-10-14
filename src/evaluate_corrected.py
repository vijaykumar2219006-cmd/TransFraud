
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

BASE = "datasets/SyntheticBanking/sequences_corrected"

mean = np.load("models/feature_mean.npy")
std = np.load("models/feature_std.npy")
std[std == 0] = 1

model = tf.keras.models.load_model(
    "models/transfraud_bilstm_transformer.keras"
)


def evaluate_split(name):
    data = np.load(f"{BASE}/{name}_sequences.npz")
    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)

    X = (X - mean) / std
    probabilities = model.predict(
        X, batch_size=512, verbose=0
    ).ravel()

    print(f"\n===== {name.upper()} =====")
    print("Transactions:", len(y))
    print("Fraud cases:", int(y.sum()))
    print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))
    print(
        "Average precision:",
        round(average_precision_score(y, probabilities), 4),
    )

    return y, probabilities


yv, pv = evaluate_split("validation")

print("\nVALIDATION THRESHOLD COMPARISON")
print("Threshold | TP | FP | FN | Precision | Recall | F1")

for threshold in [0.5, 0.7, 0.8, 0.9]:
    pred = (pv >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        yv, pred, labels=[0, 1]
    ).ravel()

    print(
        f"{threshold:.1f} | {tp} | {fp} | {fn} | "
        f"{precision_score(yv, pred, zero_division=0):.4f} | "
        f"{recall_score(yv, pred, zero_division=0):.4f} | "
        f"{f1_score(yv, pred, zero_division=0):.4f}"
    )

yt, pt = evaluate_split("test")


threshold = 0.90
predictions = (pt >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    yt, predictions, labels=[0, 1]
).ravel()

print(f"\nFINAL TEST RESULTS AT THRESHOLD {threshold}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")
print(f"Precision: {precision_score(yt, predictions, zero_division=0):.4f}")
print(f"Recall: {recall_score(yt, predictions, zero_division=0):.4f}")
print(f"F1-score: {f1_score(yt, predictions, zero_division=0):.4f}")

print("\nEvaluation complete. No model files were modified.")
