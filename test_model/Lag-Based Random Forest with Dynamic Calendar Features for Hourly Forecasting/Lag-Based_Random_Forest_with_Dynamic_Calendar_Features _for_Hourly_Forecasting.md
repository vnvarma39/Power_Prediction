This approach uses a Random Forest Regressor to forecast hourly electricity demand for the year 2023, leveraging lag-based features from the past 24 hours and dynamically updating calendar-based features such as hour of the day, day of the week, and weekend status. Unlike static models, it performs recursive forecasting by feeding each predicted value back into the model to predict the next hour, allowing it to simulate real-world conditions. The method works well due to its ability to capture both short-term dependencies and seasonal patterns using non-linear modeling, making it accurate, scalable, and interpretable.

![image](https://github.com/user-attachments/assets/c7c90d68-b120-445e-83a6-2a3a037046d4)

![image](https://github.com/user-attachments/assets/c127c867-1391-4d5f-ac6e-fba7348b76d3)

![image](https://github.com/user-attachments/assets/98d56b08-b68f-476b-ad69-b74b732094f4)

![image](https://github.com/user-attachments/assets/0bae0829-7e8b-4ea1-9f08-4ad2e5aaae8a)

