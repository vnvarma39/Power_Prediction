![image](https://github.com/user-attachments/assets/145dced8-f3b7-402a-8d1d-a3da5544be6c)

![image](https://github.com/user-attachments/assets/4c82f7d6-8576-414f-9575-7c71de75ee77)

This model uses an autoregressive time series forecasting approach, where the electricity demand for the next hour is predicted using the demand values from the past 24 hours. It employs a sliding window technique to create lagged features, which are then flattened and fed into a Random Forest Regressor—a tree-based ensemble learning model capable of capturing nonlinear patterns in the data. The model is trained and evaluated on a continuous sequence of hourly demand data without any shuffling, ensuring temporal integrity. This method is simple, robust, and effective for short-term forecasting tasks where sequential dependencies are captured through historical feature engineering rather than sequence models like LSTMs.

✅ Why this method works well:
Doesn’t require sequence models like LSTMs.

Works with structured tabular data.

Robust to noise and nonlinear trends.
