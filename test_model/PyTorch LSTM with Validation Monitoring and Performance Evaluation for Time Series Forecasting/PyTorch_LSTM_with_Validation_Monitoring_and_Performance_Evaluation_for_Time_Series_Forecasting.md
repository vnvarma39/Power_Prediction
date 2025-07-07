This model uses a deep LSTM architecture implemented in PyTorch to predict hourly electricity demand, training on the past 24 hours of scaled demand values to forecast the next hour. It incorporates an 80-10-10 train-validation-test split to ensure generalization and prevent overfitting. The model is trained over 70 epochs with batch learning and dropout regularization, with both training and validation losses tracked throughout. After training, the model is evaluated on a test set using MAE, RMSE, and R² metrics, and then used to recursively forecast all 8760 hours of 2023. The predictions are inverse-scaled and saved as a CSV file.

![image](https://github.com/user-attachments/assets/97b2b185-aeae-48a2-9c17-42cbcf91e296)

![image](https://github.com/user-attachments/assets/425f6d54-f995-4edc-9287-074afd415545)

![image](https://github.com/user-attachments/assets/8354f620-200c-467c-b379-b85a6eeeabe8)

![image](https://github.com/user-attachments/assets/c1ad6c96-5136-43bd-b2c5-2da0cc729176)

![image](https://github.com/user-attachments/assets/15994191-2e1e-40c3-b5e5-144345ff958a)

![image](https://github.com/user-attachments/assets/049792af-46e8-4562-9740-839e93922751)


This version is the most robust and production-ready among the models implemented so far
