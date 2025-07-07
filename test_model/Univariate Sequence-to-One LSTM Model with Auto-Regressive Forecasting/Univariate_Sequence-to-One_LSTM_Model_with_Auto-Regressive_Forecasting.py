import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

# -------------------------------
# Step 1: Load and Clean Data
# -------------------------------

# Load CSV file
file_path = 'merge-csv.com__685e341ae0fa0.csv'
df = pd.read_csv(file_path, header=None)

# Assign column names
columns = ['Date'] + [f'Hour_{i}' for i in range(1, 25)]
df.columns = columns

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)

# Melt to long format
df_long = df.melt(id_vars='Date', var_name='Hour', value_name='Demand')
df_long['Hour'] = df_long['Hour'].str.replace('Hour_', '').astype(int)
df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Hour'], unit='h')
df_long.set_index('DateTime', inplace=True)
df_long = df_long[['Demand']].sort_index()

# Convert to numeric and interpolate missing values
df_long['Demand'] = pd.to_numeric(df_long['Demand'], errors='coerce')
df_long = df_long.interpolate(method='time')

# Normalize the data
scaler = MinMaxScaler()
df_long['Demand_Scaled'] = scaler.fit_transform(df_long[['Demand']])

# -------------------------------
# Step 2: Create Sequences for LSTM
# -------------------------------

def create_sequences(data, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24
data_scaled = df_long['Demand_Scaled'].values
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split training data
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input to [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# -------------------------------
# Step 3: Build LSTM Model Using Keras
# -------------------------------

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# -------------------------------
# Step 4: Predict 2023 Hourly Demand
# -------------------------------

# Generate date range for 2023
future_dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='H')

# Start with last known sequence from historical data
current_sequence = data_scaled[-SEQ_LENGTH:].copy()
predictions = []

for _ in future_dates:
    # Prepare input
    X_input = current_sequence.reshape((1, SEQ_LENGTH, 1))
    
    # Make prediction
    pred = model.predict(X_input, verbose=0)[0][0]
    predictions.append(pred)
    
    # Update sequence for next prediction
    current_sequence = np.roll(current_sequence, shift=-1)
    current_sequence[-1] = pred

# Inverse transform predictions
pred_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# -------------------------------
# Step 5: Save Predictions to CSV
# -------------------------------

pred_df = pd.DataFrame({
    'DateTime': future_dates,
    'Predicted_Demand': pred_actual
})

pred_df.to_csv('2023_lstm_predictions.csv', index=False)
print("✅ Predictions saved to '2023_lstm_predictions.csv'")
