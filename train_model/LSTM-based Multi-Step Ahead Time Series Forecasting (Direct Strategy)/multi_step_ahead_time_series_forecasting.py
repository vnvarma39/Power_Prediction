import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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
df_long['DateTime'] = pd.to_datetime(df_long['Date']) + pd.to_timedelta(df_long['Hour'], unit='h')
df_long.set_index('DateTime', inplace=True)
df_long = df_long[['Demand']].sort_index()

# Convert to numeric and interpolate missing values
df_long['Demand'] = pd.to_numeric(df_long['Demand'], errors='coerce')
df_long = df_long.interpolate(method='time')

# Normalize data
scaler = MinMaxScaler()
df_long['Demand_Scaled'] = scaler.fit_transform(df_long[['Demand']])

# -------------------------------
# Step 2: Create Sequences for Multi-Step Forecasting
# -------------------------------

def create_sequences(data, seq_length=24, pred_steps=6):
    X, y = [], []
    data = data.values
    for i in range(len(data) - seq_length - pred_steps + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_steps])
    return np.array(X), np.array(y)

SEQ_LENGTH = 24   # Use past 24 hours
PRED_STEPS = 6    # Predict next 6 hours

X, y = create_sequences(df_long['Demand_Scaled'], SEQ_LENGTH, PRED_STEPS)

# Split into train/test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, timesteps, features)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -------------------------------
# Step 3: Define LSTM Model for Multi-Step Forecasting
# -------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PRED_STEPS):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
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
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
for epoch in range(EPOCHS):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}")

# -------------------------------
# Step 5: Evaluate and Plot Results
# -------------------------------

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).numpy()  # No need to call .numpy()

# Inverse transform predictions
test_preds_actual = scaler.inverse_transform(test_preds)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_actual.flatten(), test_preds_actual.flatten()))
mape = mean_absolute_percentage_error(y_test_actual.flatten(), test_preds_actual.flatten()) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot first few test samples
plt.figure(figsize=(12, 6))
for i in range(5):  # Plot 5 test samples
    plt.subplot(5, 1, i+1)
    plt.plot(np.arange(PRED_STEPS), y_test_actual[i], label='Actual')
    plt.plot(np.arange(PRED_STEPS), test_preds_actual[i], label='Predicted')
    plt.title(f'Sample {i+1}')
    plt.xlabel('Future Hour')
    plt.ylabel('Demand')
    plt.legend()
plt.tight_layout()
plt.show()
