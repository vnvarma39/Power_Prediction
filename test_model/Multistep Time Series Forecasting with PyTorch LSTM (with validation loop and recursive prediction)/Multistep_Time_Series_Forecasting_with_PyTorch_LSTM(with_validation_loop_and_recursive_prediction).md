This model uses a deep LSTM (Long Short-Term Memory) neural network implemented in PyTorch to forecast hourly electricity demand for the year 2023. It leverages the last 24 hours of normalized demand data to predict the next hour, training on historical data using an 80-10-10 split for training, validation, and testing. The model is trained over 50 epochs using mini-batches and includes dropout regularization to prevent overfitting. Once trained, the model is rolled forward in a recursive manner to generate demand forecasts hour-by-hour for the entire year, using its own predictions as input for the next steps. Finally, predictions are inverse-scaled back to the original demand scale and saved as a CSV.

![image](https://github.com/user-attachments/assets/06639187-4509-4460-9543-09d14f5ec082)

![image](https://github.com/user-attachments/assets/2bf7d392-52a4-4959-9f64-ada0f718d71a)

![image](https://github.com/user-attachments/assets/e985031f-a3d9-4127-a0e4-d882e055ce56)


