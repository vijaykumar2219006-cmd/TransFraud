import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, MultiHeadAttention,
    LayerNormalization, Dense, Dropout, GlobalAveragePooling1D
)

# =========================
# 1. Load sequences
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
# 3. BiLSTM + Transformer
# =========================

inputs = Input(shape=(10, 12))

# BiLSTM
x = Bidirectional(
    LSTM(64, return_sequences=True)
)(inputs)

x = Dropout(0.3)(x)

# Transformer Self-Attention
attention = MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(x, x)

# Residual connection + normalization
x = LayerNormalization()(x + attention)

# Feed-forward network
ff = Dense(128, activation="relu")(x)
ff = Dropout(0.3)(ff)
ff = Dense(128)(ff)

# Residual connection + normalization
x = LayerNormalization()(x + ff)

# Convert sequence to single representation
x = GlobalAveragePooling1D()(x)

# Classification
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# =========================
# 4. Compile
# =========================

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
# 5. Class weights
# =========================

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)

# =========================
# 6. Train
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
# 7. Save model
# =========================

model.save("transfraud_bilstm_transformer.keras")

print("\nBiLSTM + Transformer training complete!")
print("Model saved as transfraud_bilstm_transformer.keras")