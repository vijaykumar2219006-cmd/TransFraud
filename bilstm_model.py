import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

# =========================
# 1. Load data
# =========================

train = np.load(
    "datasets/SyntheticBanking/sequences/train_sequences.npz"
)

val = np.load(
    "datasets/SyntheticBanking/sequences/validation_sequences.npz"
)

X_train = train["X"].astype("float32")
y_train = train["y"]

X_val = val["X"].astype("float32")
y_val = val["y"]

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# =========================
# 2. Feature scaling
# =========================

mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# =========================
# 3. Handle class imbalance
# =========================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

# =========================
# 4. Build BiLSTM
# =========================

model = Sequential([
    Input(shape=(10, 12)),

    Bidirectional(
        LSTM(64, return_sequences=True)
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(32)
    ),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# =========================
# 5. Train
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=256,
    class_weight=class_weights
)

# =========================
# 6. Save model
# =========================

model.save("bilstm_model.keras")

# Save scaling values for later
np.save("feature_mean.npy", mean)
np.save("feature_std.npy", std)

print("\nBiLSTM training complete!")
print("Model saved as bilstm_model.keras")