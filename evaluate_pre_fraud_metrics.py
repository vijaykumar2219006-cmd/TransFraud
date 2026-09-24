import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("pre_fraud_correct_results.csv")

y_true = df["actual_label"]
y_prob = df["fraud_probability"]

print("Number of fraud cases:", len(df))

# ROC-AUC
auc = roc_auc_score(y_true, y_prob)

print("\nROC-AUC:", round(auc, 4))

# Evaluate thresholds
for threshold in [0.50, 0.95]:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n==============================")
    print("Threshold:", threshold)
    print("==============================")

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)