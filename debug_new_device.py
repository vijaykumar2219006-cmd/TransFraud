import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/transfraud_bilstm_transformer.keras"
MEAN_PATH = "models/feature_mean.npy"
STD_PATH = "models/feature_std.npy"

model = load_model(MODEL_PATH, compile=False)
feature_mean = np.load(MEAN_PATH)
feature_std = np.load(STD_PATH)

print("new_device mean:", feature_mean.flatten()[6])
print("new_device std:", feature_std.flatten()[6])

# Raw sequence copied from the API debug output.
sequence = np.array([
    [38.47, 0, 0, 0, 0, 0, 1, 0, -1, 20, 2, 10],
    [90.57, 1, 38.47, 2.3543, 0, 1, 0, 28, 14773501, 20, 5, 4],
    [34.55, 2, 64.52, 0.5355, 1, 2, 0, 27, 1128961, 22, 4, 5],
    [5.96, 3, 54.53, 0.1093, 2, 3, 0, 37, 5229808, 11, 2, 7],
    [8.49, 4, 42.3875, 0.2003, 1, 4, 0, 39, 216922, 23, 4, 7],
    [8.55, 5, 35.608, 0.2401, 0, 5, 0, 38, 1983046, 22, 6, 7],
    [60.59, 6, 31.0983, 1.9483, 2, 6, 0, 56, 7095379, 1, 5, 10],
    [100, 7, 35.3114, 2.8319, 3, 0, 1, 0, 3606427, 18, 4, 11]
], dtype=np.float32)

# Pad to the required sequence length of 10.
padding = np.zeros((2, 12), dtype=np.float32)
sequence = np.vstack([padding, sequence])

def predict(raw_sequence):
    scaled = (raw_sequence - feature_mean) / feature_std

    # Ensure the model receives exactly (1, 10, 12)
    batch = scaled.reshape(1, 10, 12)

    print("Input shape:", batch.shape)
    return float(model.predict(batch, verbose=0)[0][0])

# Baseline: new_device = 1
p_new = predict(sequence)

# Controlled test: change only the final row's new_device value.
sequence_existing = sequence.copy()
sequence_existing[-1, 6] = 0
p_existing = predict(sequence_existing)

print(f"Probability with new_device=1: {p_new:.6f}")
print(f"Probability with new_device=0: {p_existing:.6f}")
print(f"Difference: {p_new - p_existing:+.6f}")

# Test merchant history independently.
sequence_merchant = sequence.copy()

# Restore the new-device value to 0.
sequence_merchant[-1, 6] = 0

# Change only merchant_previous_count (feature index 7).
sequence_merchant[-1, 7] = 56

p_merchant = predict(sequence_merchant)

print(
    "Probability with new_device=0 and merchant_history=56:",
    f"{p_merchant:.6f}"
)

# Isolate device_previous_count.
sequence_device_count = sequence.copy()

# Keep new_device = 0.
sequence_device_count[-1, 6] = 0

# Change only device_previous_count (feature index 5).
sequence_device_count[-1, 5] = 7

p_device_count = predict(sequence_device_count)

print(
    "Probability with device history count = 7:",
    f"{p_device_count:.6f}"
)