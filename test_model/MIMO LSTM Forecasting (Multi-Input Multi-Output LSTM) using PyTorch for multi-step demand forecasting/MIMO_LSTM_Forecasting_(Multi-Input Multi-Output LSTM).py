import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and Preprocess Data
# -------------------------------

file_path = 'merge-csv.com__685e341ae0fa0.csv'
df = pd.read_csv(file_path, header=None)

columns = ['Date'] + [f'Hour_{i}' for i in range(1, 25)]
df.columns = columns

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).reset_index(drop=True)

df_long = df.melt(id_vars='Date', var_name='Hour', value_name='Demand')
df_long['Hour'] = df_long['Hour'].str.replace('Hour_', '').astype(int)
df_long['DateTime'] = df_long['Date'] + pd.to_timedelta(df_long['Hour'] - 1, unit='h')
df_long.set_index('DateTime', inplace=True)
df_long = df_long[['Demand']].sort_index()

df_long['Demand'] = pd.to_numeric(df_long['Demand'], errors='coerce')
df_long = df_long.interpolate(method='time')

scaler = MinMaxScaler()
df_long['Demand_Scaled'] = scaler.fit_transform(df_long[['Demand']])

# -------------------------------
# Step 2: Create Sequences (MIMO Version)
# -------------------------------

def create_sequences_mimo(data, seq_length=24, pred_steps=6):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_steps + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+pred_steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24
PRED_STEPS = 6

X, y = create_sequences_mimo(df_long['Demand_Scaled'].values, SEQ_LENGTH, PRED_STEPS)

split_train_val = int(len(X) * 0.8)
split_val_test = int(len(X) * 0.9)

X_train, X_val, X_test = X[:split_train_val], X[split_train_val:split_val_test], X[split_val_test:]
y_train, y_val, y_test = y[:split_train_val], y[split_train_val:split_val_test], y[split_val_test:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -------------------------------
# Step 3: Define LSTM Model (MIMO Strategy)
# -------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=6, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])
        return out

model = LSTMModel(output_size=PRED_STEPS)
print(model)

# -------------------------------
# Step 4: Train the Model (with Validation)
# -------------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 70
BATCH_SIZE = 32

train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            y_val_pred = model(x_val_batch)
            val_loss += criterion(y_val_pred, y_val_batch).item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Step 5: Evaluate on Test Set
# -------------------------------

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor).numpy()

test_preds_actual = scaler.inverse_transform(test_preds)
y_test_actual = scaler.inverse_transform(y_test_tensor.numpy())

test_preds_flat = test_preds_actual.flatten()
y_test_flat = y_test_actual.flatten()

mae = mean_absolute_error(y_test_flat, test_preds_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, test_preds_flat))
r2 = r2_score(y_test_flat, test_preds_flat)

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R² Score: {r2:.2f}")

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(y_test_actual[i], label='Actual')
    plt.plot(test_preds_actual[i], label='Predicted')
    plt.title(f"Sample {i+1}")
    plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# ✅ Step 6: Fixed MIMO Forecast for 2023
# -------------------------------

future_dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='H')
current_sequence = df_long['Demand_Scaled'].values[-SEQ_LENGTH:].copy()
predictions = []
predicted_timestamps = []

i = 0
while i < len(future_dates):
    X_input = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred_scaled = model(X_input).squeeze().numpy()

    predictions.extend(pred_scaled)
    predicted_timestamps.extend(future_dates[i:i+PRED_STEPS])

    for p in pred_scaled:
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pa

    i += PRED_STEPS

pred_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

pred_df = pd.DataFrame({
    'DateTime': predicted_timestamps,
    'Predicted_Demand': pred_actual
})

pred_df.to_csv('2023_lstm_mimo_predictions.csv', index=False)
print("✅ Predictions saved to '2023_lstm_mimo_predictions.csv'")
