This model implements a Multi-Input Multi-Output (MIMO) LSTM neural network to forecast electricity demand six hours ahead based on the previous 24 hours. The time series data is preprocessed by interpolating missing values and scaled using MinMax normalization. The model uses a 2-layer LSTM followed by a dropout layer and a dense layer to output 6 continuous hourly predictions simultaneously. It is trained on historical data with validation and test splits to monitor generalization. Once trained, it performs recursive multi-step forecasting for all of 2023 using a sliding window over the predicted values. The approach is efficient for handling correlated multi-step outputs and avoids error accumulation typical of recursive single-output LSTM models.

![image](https://github.com/user-attachments/assets/4a9bac3c-2de6-434d-b67a-0944aa802823)

![image](https://github.com/user-attachments/assets/90ed7688-2e5e-491d-b62f-4b399bdae9cc)


![image](https://github.com/user-attachments/assets/e159f436-6d4a-4846-aeff-e1701fde1f1f)

![image](https://github.com/user-attachments/assets/d3ac5362-08d1-43d8-96c0-97b8d6c32eca)

![image](https://github.com/user-attachments/assets/e1aee691-b3e3-4e87-ac47-128c3a1795a6)

![image](https://github.com/user-attachments/assets/dc2f9916-09bc-4ebb-aec1-ea386185c84f)
