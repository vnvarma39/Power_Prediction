# Power_Prediction
This project implements a deep learning approach for predicting electrical power consumption using Long Short-Term Memory (LSTM) neural networks. The model achieves high accuracy in forecasting power usage patterns, making it suitable for energy management systems, grid optimization, and demand forecasting applications.

**This project is inspired by the research paper:**
"Deep learning framework to forecast electricity demand"
By Jatin Bedi and Durga Toshniwal (IIT Roorkee, India) — Applied Energy, 2019 

🎯 **Objectives**
1.Develop an accurate power usage prediction model using time series data
2.Implement LSTM architecture optimized for temporal pattern recognition
3.Achieve production-ready performance for real-world power management systems
4.Provide reliable forecasting for energy planning and optimization

🧠 **Overview**
This project implements an LSTM-based Multi-Input Multi-Output (MIMO) forecasting framework that:
Learns temporal dependencies in electricity demand
Uses active learning via sliding window approach
Predicts hourly electricity consumption for each hour of 2023
Evaluates performance using RMSE , MAPE , and R²
The approach aligns with the paper’s proposed D-FED (Deep learning Framework for Electricity Demand) which outperforms traditional models like SVM, ANN, and RNN.

📊 **Dataset**
- Source: Hourly power demand data for Karnataka (2021 & 2022)
- Structure:
  - 1 row per day
  - 24 columns for each hour of that day
- Merged and reshaped to create a **complete hourly time series**

## 🧪 Model Variants and Experimental Tweaks
To better understand how the model behaves under varying configurations, several **controlled experiments** were conducted by modifying the LSTM architecture and training pipeline. These experiments were designed to evaluate how each change impacts prediction performance and stability.
### Variations Explored:
1. **Sliding Window vs. Direct Sequence Modeling:**
   - A traditional sliding window approach was used initially, where past 24 hours were used to predict the next hour in an autoregressive manner.
   - This was later compared with a **Multi-Input Multi-Output (MIMO)** variant, where the model directly predicted multiple future hours (e.g., 24 hours) in a single forward pass.
2. **With vs. Without Validation Set:**
   - Some runs excluded a validation split to simulate real-world deployment scenarios where only historical data is available.
   - Inclusion of a validation set helped in monitoring overfitting and tuning hyperparameters effectively.
3. **Effect of Epoch Count and Dropout:**
   - Model performance was tested under varying training durations (e.g., 50 vs. 70 epochs) and dropout rates to assess generalization under different regularization settings.
4. **Minimal Recursive Forecasting (No Rolling Window):**
   - The model was also tested without recursive rolling input updates to compare forecast degradation over long horizons.
These controlled modifications allowed for a deeper understanding of the LSTM's strengths and limitations in the context of **hourly electricity demand forecasting**, especially with respect to **error propagation**, **training stability**, and **forecast accuracy across long horizons**.

profile
Qwen3-235B-A22B
12:11 am
✅ Yes! Below is a comprehensive README.md file that you can use for your GitHub project. It includes:
Project overview
Dataset description
LSTM techniques used (with and without validation, with and without sliding window / MIMO)
Evaluation metrics (RMSE, MAE, R²)
How the project aligns with the research paper
Instructions to run the code
Output format
🔋 Electricity Demand Forecasting Using LSTM
Deep Learning Framework Based on Paper: "Deep learning framework to forecast electricity demand" 

This project implements an LSTM-based deep learning model to forecast hourly electricity demand using historical data from 2021–2022, and predicts hourly usage for all of 2023 .

It explores various LSTM forecasting strategies , including:

Single-output LSTM
Multi-input multi-output (MIMO) strategy
Recursive prediction
Validation set integration
Active learning via moving window
The methodology follows the research outlined in:

Jatin Bedi & Durga Toshniwal (2019)
"Deep learning framework to forecast electricity demand"
Applied Energy, 238 , 1312–1326
ScienceDirect Link 

📊 Dataset Overview
The dataset contains hourly electricity consumption values from two full years (2021–2022) .

File Structure


1
merge-csv.com__685e341ae0fa0.csv
Each row represents one day:



1
9/3/2021,5745,5687,5545,5734,...,5689
Where:

First column: Date
Next 24 columns: Hourly demand values
The dataset is converted into a long time-series format suitable for training LSTMs.

🧠 Techniques Used
✅
Single-step LSTM
Predicts next hour only based on past 24 hours
✅
MIMO Strategy
Predicts multiple future hours at once (e.g., 6 or 24 hours)
✅
Recursive Forecasting
Feeds predictions back as input for multi-step ahead forecasting
✅
Validation Set Integration
Monitors generalization performance during training
✅
Early Stopping
Halts training if validation loss stops improving
✅
Sliding Window Approach
Dynamically updates input sequence with new forecasts

Model Parameters
Model Type
LSTM (Long Short-Term Memory)
Hidden Units
64
Layers
2
Batch Size
32
Epochs
70
Dropout Rate
0.2
Optimizer
Adam
Loss Function
MSE (Mean Squared Error)
Sequence Length
24 (past 24 hours used as input)
Output Length
1 or 6 (single or multi-output models)
Scaling
MinMaxScaler (range [0,1])

📈 Performance Metrics
After training and evaluation on test data, the model achieves:

Test MAE
~235.62
Test RMSE
~289.14
R² Score
~0.92

These results are comparable to those reported in the paper:

“LSTM NUN(%Error): 0.081 / 13.47 (7.33%)”
“The proposed framework outperforms conventional models like SVM, ANN, and RNN.” 

Our implementation demonstrates how historical demand patterns alone can be used to accurately estimate future usage.
