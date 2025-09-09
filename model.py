import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------
# Load and preprocess data
# ------------------------
file_path = "Final.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")

X = data.drop(columns=["rally_id", "target_x", "target_y"]).values
y = data[["target_x", "target_y"]].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape to (samples, timesteps, features)
# Each rally has 3 shots (Djokovic + opponent + serve) => 3 timesteps
# At each timestep, we have 6 features (dj_x, dj_y, op_x, op_y, sp_x, sp_y)
X_seq = X_scaled.reshape(X_scaled.shape[0], 3, 6)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

# ------------------------
# Build CNN + LSTM model
# ------------------------
model = Sequential()

# CNN layer to extract local patterns
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 6)))
model.add(MaxPooling1D(pool_size=1))

# LSTM layer to capture sequence
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

# Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output: 2 values (target_x, target_y)
model.add(Dense(2))

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ------------------------
# Train model
# ------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

# ------------------------
# Evaluate
# ------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE:", loss)
print("Test MAE:", mae)

# Example predictions
preds = model.predict(X_test[:5])
print("True values:\n", y_test[:5])
print("Predicted values:\n", preds)
