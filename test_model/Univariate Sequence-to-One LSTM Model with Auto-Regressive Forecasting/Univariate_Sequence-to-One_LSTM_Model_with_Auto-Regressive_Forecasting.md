This approach employs a Long Short-Term Memory (LSTM) neural network to model and forecast hourly electricity demand. It transforms historical demand data into 24-hour sliding windows to capture temporal patterns and daily cycles. After training, it uses an auto-regressive strategy where each new prediction is appended to the input sequence to forecast the next hour, allowing the model to produce a full year’s worth of hourly predictions. The use of LSTM ensures the model captures time-based dependencies effectively, and normalization along with dropout helps improve generalization and stability.

![image](https://github.com/user-attachments/assets/fc81020a-9207-4058-8796-8b1cbc7244eb)


![image](https://github.com/user-attachments/assets/db187546-aaf2-49ce-afc7-a0a07500efe3)
