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
