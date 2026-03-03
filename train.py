import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import load_and_preprocess_data
from model import create_sequences, build_model

# 1️⃣ Load data
data, scaler = load_and_preprocess_data()

sequence_length = 10
X, y = create_sequences(data, sequence_length)

# 2️⃣ Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 3️⃣ Build model
model = build_model((sequence_length, 2))

# 4️⃣ Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# 5️⃣ Predict
y_pred = model.predict(X_test)

# 6️⃣ Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)

# 7️⃣ Plot loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 8️⃣ Future Prediction (20 steps)
last_sequence = data[-sequence_length:]
future_points = []

current_seq = last_sequence.reshape(1, sequence_length, 2)

for _ in range(20):
    next_point = model.predict(current_seq)[0]
    future_points.append(next_point)

    current_seq = np.append(current_seq[:, 1:, :],
                            [[next_point]],
                            axis=1)

future_points = np.array(future_points)

# Convert back to real GPS coordinates
future_real = scaler.inverse_transform(future_points)

print("\nNext 20 Predicted GPS Coordinates:")
print(future_real)

# 9️⃣ Save model
model.save("trajectory_model.h5")
print("\nModel saved successfully.")