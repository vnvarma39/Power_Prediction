import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# Step 1: Load and Preprocess Data
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
# Step 2: Create Sequences
# -------------------------------

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24  # Use past 24 hours to predict next hour
X, y = create_sequences(df_long['Demand_Scaled'].values, SEQ_LENGTH)

# Train-validation-test split
split_train_val = int(len(X) * 0.8)
split_val_test = int(len(X) * 0.9)
X_train, X_val, X_test = X[:split_train_val], X[split_train_val:split_val_test], X[split_val_test:]
y_train, y_val, y_test = y[:split_train_val], y[split_train_val:split_val_test], y[split_val_test:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, timesteps, features)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -------------------------------
# Step 3: Define LSTM Model
# -------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])  # Take last timestep output
        return out

model = LSTMModel()
print(model)

# -------------------------------
# Step 4: Train the Model
# -------------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 32

# Convert to DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            y_val_pred = model(x_val_batch)
            val_loss = criterion(y_val_pred.squeeze(), y_val_batch)
            val_losses.append(val_loss.item())
    
    avg_val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, Val Loss: {avg_val_loss:.5f}")

# -------------------------------
# Step 5: Forecast 2023 Hourly Demand
# -------------------------------

# Generate date range for 2023
future_dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='H')

# Start with last known sequence from historical data
current_sequence = df_long['Demand_Scaled'].values[-SEQ_LENGTH:].copy()
predictions = []

for _ in future_dates:
    # Prepare input
    X_input = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # shape: (1, seq_len, 1)
    
    # Make prediction
    pred = model(X_input).item()
    predictions.append(pred)
    
    # Update sequence
    current_sequence = np.roll(current_sequence, shift=-1)
    current_sequence[-1] = pred

# Inverse transform predictions
pred_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# -------------------------------
# Step 6: Save Predictions to CSV
# -------------------------------

# Create DataFrame
pred_df = pd.DataFrame({
    'DateTime': future_dates,
    'Predicted_Demand': pred_actual
})

# Save to CSV
pred_df.to_csv('2023_lstm_original_2_predictions.csv', index=False)
print("✅ Predictions saved to '2023_lstm_original_2_predictions.csv'")
