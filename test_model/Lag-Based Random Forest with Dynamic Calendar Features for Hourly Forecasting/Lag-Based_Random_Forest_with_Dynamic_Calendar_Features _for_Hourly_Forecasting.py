import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load and Clean Data
# -------------------------------

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

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------

def create_features(data):
    data = data.copy()
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['WeekOfYear'] = data.index.isocalendar().week.astype(int)
    data['HourOfDay'] = data.index.hour
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    return data

def add_lags(data, lags=24):
    df_ = data.copy()
    for lag in range(1, lags+1):
        df_[f'Demand_Lag_{lag}'] = df_['Demand'].shift(lag)
    df_.dropna(inplace=True)
    return df_

df_featured = create_features(df_long)
df_featured = add_lags(df_featured, lags=24)

X = df_featured.drop(columns=['Demand'])
y = df_featured['Demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------------------
# Step 3: Train Model
# -------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
test_preds = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, test_preds) * 100
print(f"✅ Model MAPE on test data: {mape:.2f}%")

# -------------------------------
# Step 4: Feature Importance
# -------------------------------

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 5: Forecast 2023 Hourly Demand
# -------------------------------

future_dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00', freq='H')
current_data = X.iloc[-1].copy()
predictions = []

for date in future_dates:
    # Update date-based features
    current_data['HourOfDay'] = date.hour
    current_data['DayOfWeek'] = date.dayofweek
    current_data['Month'] = date.month
    current_data['WeekOfYear'] = date.isocalendar()[1]
    current_data['IsWeekend'] = int(date.dayofweek >= 5)

    # Predict
    X_input = current_data.values.reshape(1, -1)
    pred = model.predict(X_input)[0]
    predictions.append(pred)

    # Update lag features
    for lag in range(24, 0, -1):
        if lag == 1:
            current_data[f'Demand_Lag_{lag}'] = pred
        else:
            current_data[f'Demand_Lag_{lag}'] = current_data[f'Demand_Lag_{lag - 1}']

# -------------------------------
# Step 6: Save Predictions
# -------------------------------

pred_df = pd.DataFrame({
    'DateTime': future_dates,
    'Predicted_Demand': predictions
})

pred_df.to_csv('2023_hourly_predictions2.csv', index=False)
print("📦 Hourly predictions saved to '2023_hourly_predictions2.csv'")
