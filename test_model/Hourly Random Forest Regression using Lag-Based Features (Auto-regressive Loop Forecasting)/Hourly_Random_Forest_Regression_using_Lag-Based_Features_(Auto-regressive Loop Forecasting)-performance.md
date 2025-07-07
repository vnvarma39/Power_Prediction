This approach forecasts hourly electricity demand for 2023 using a Random Forest regression model trained on historical demand and engineered time-based features. It captures past patterns using a 24-hour lag window, and incorporates calendar information like hour of day, weekday/weekend, and month. Forecasting is done recursively: each new predicted value is used as input for predicting the next hour, simulating a real-world streaming scenario. This method is powerful for short-term high-resolution forecasting due to its ability to model complex, non-linear dependencies without requiring deep learning models.

![image](https://github.com/user-attachments/assets/3cdf6e24-94c0-4c40-9e8c-61fdd3ec4dfd)

![image](https://github.com/user-attachments/assets/0368464c-06cc-48f2-8bc7-e5247bbda358)


![image](https://github.com/user-attachments/assets/66d67920-6ec0-479d-b7fc-26e77b6a7e80)
