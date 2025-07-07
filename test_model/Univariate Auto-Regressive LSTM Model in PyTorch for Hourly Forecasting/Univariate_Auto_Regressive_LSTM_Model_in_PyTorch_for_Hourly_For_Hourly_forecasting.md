This approach uses a Long Short-Term Memory (LSTM) network built in PyTorch to forecast hourly electricity demand. It transforms historical demand data into a time series of 24-hour sequences (lags) and uses these to predict the next hour’s demand. After training on 80% of the data, the model forecasts hourly demand for the entire year of 2023 by auto-regressively feeding each predicted value back into the input sequence. The demand values are normalized before training and inverse-transformed after prediction for interpretability. The model captures temporal trends effectively, making it well-suited for time-series forecasting tasks like this.

![image](https://github.com/user-attachments/assets/470325ad-2ee1-4b09-8d6d-fcbc679fe858)
![image](https://github.com/user-attachments/assets/6e2842b4-39ad-42dc-acb0-f7c91ff53be3)

![image](https://github.com/user-attachments/assets/9fe65b6a-1448-433b-9cb1-7d411346a927)

![image](https://github.com/user-attachments/assets/a877ffca-9522-4cee-838a-cb67280869b1)

![image](https://github.com/user-attachments/assets/764c4cc6-950d-4a44-9381-11f77cfad7bf)
