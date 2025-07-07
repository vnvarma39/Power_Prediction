This approach uses a Random Forest Regressor to predict hourly electricity demand by transforming the time series data into a supervised learning problem. It constructs lag features for the past 24 hours and adds calendar/time-based features such as day of the week, month, hour of the day, and weekend indicators. The model is trained on historical hourly data and is then used to recursively forecast demand for every hour in 2023, using each new prediction as input for the next step.

![image](https://github.com/user-attachments/assets/a19cef01-cb02-443d-87b6-240578eb7a70)
![image](https://github.com/user-attachments/assets/043874d8-4a95-4245-82ac-7f97cf816b15)

![image](https://github.com/user-attachments/assets/b5e2e89c-cbb9-47ef-9ee6-0547e42e038e)


![image](https://github.com/user-attachments/assets/690da387-b0dd-4906-af3c-25235ad6ae1a)
