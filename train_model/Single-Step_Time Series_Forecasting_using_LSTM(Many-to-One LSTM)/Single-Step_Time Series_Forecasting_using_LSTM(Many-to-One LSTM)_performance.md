This approach uses a Many-to-One LSTM model to forecast electricity demand one step (hour) ahead based on the previous 24 hourly values. The raw time series data is first cleaned, interpolated, and normalized. A sliding window of 24 time steps (representing past 24 hours) is used to construct input sequences, where each sequence predicts the next hour’s demand. These sequences are passed into an LSTM network with two hidden layers, followed by a linear layer to output a single predicted value. The model is trained using Mean Squared Error (MSE) loss and evaluated using RMSE and MAPE.


![image](https://github.com/user-attachments/assets/9c87bd4c-6265-4817-bd59-b717d4557024)

![image](https://github.com/user-attachments/assets/f05b469d-5867-495d-b781-6ee94f81da0a)
