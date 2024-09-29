# Stock Performance Prediction Using Bidirectional LSTM
This project focuses on predicting the stock performance (specifically the closing price) of Company "X" using deep learning techniques. The dataset provided contains time series data, and this project will utilize techniques such as Bidirectional LSTMs and a Sequential Self-Attention mechanism to build and enhance predictive models. The task aims to develop an effective architecture for predicting stock prices using historical data.

## Project Objectives
1. **Initial Dataset Exploration and Preprocessing:**
   - Explore the dataset to identify any inherent challenges, such as missing data or irregularities in the time series.
   - The dataset primarily contains 'Date' and 'Close' columns. The 'Date' column is converted to a format suitable for time series analysis, while the 'Close' column is used as the target variable.
   - Preprocess the time series data into input-output segments with a window size of 5 (past 5 days' data) and a horizon of 1 (predicting the next day's closing price).
2. **Dataset Split:**

   Split the dataset into three parts with the following ratio:
      - 80% Training set
      - 10% Testing set
      - 10% Validation set
3. **Base Architecture: Bidirectional LSTM:**
   - Implement a baseline deep learning architecture using Bidirectional Long Short-Term Memory (LSTM) networks. LSTMs are widely known for their effectiveness in capturing patterns in time series data due to their ability to retain long-term dependencies.
   - Visualize and track the model's performance during training and validation.
4. **Enhanced Architecture: Sequential Self-Attention Mechanism:**
    - Add a **Sequential Self-Attention Mechanism** to the base Bidirectional LSTM architecture. Self-attention allows the model to focus on different parts of the time series data, giving more weight to important historical events and reducing the influence of irrelevant data points.
    - Provide an explanation of how the sequential self-attention mechanism works and how it enhances the model’s performance compared to the baseline Bidirectional LSTM model.
5. **Model Evaluation:**
    - Evaluate the performance of both architectures (Bidirectional LSTM and LSTM with Self-Attention) on the test set using appropriate evaluation metrics, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R Squared (R2). These metrics are chosen because they provide insights into the model’s ability to predict continuous values in a time series context.
    - Provide detailed explanations of the evaluation results and the model's effectiveness in predicting the stock prices.
6. **Comparison of Model Predictions:**
    - Generate a line chart to compare the actual stock prices with the predictions made by both models (Bidirectional LSTM and LSTM with Self-Attention).
    - Analyze and discuss the differences between the two models in terms of accuracy and ability to track actual stock performance.

## Dataset Description
The dataset, stored as `X.csv`, contains the following key columns:
- Date: The trading day of the stock market.
- Close: The closing price of the stock on that specific day.

## Project Workflow
1. **Data Exploration and Preprocessing:**
    - Load the dataset and handle any missing or erroneous values.
    - Convert time series data into supervised learning format (sliding window of size 5).
2. **Data Split**

   Split the preprocessed dataset into training, validation, and test sets following an 80:10:10 ratio, maintaining the chronological order of the data.
   
4. **Base Architecture (Bidirectional LSTM):**
    - Implement the base model using Bidirectional LSTM layers to capture both past and future dependencies in the time series data.
    - Train the model and evaluate its performance using metrics such as MAE, MSE, and R2.
    
5. **Enhanced Architecture (Sequential Self-Attention):**
    - Add a Sequential Self-Attention mechanism to the Bidirectional LSTM model to improve the model's ability to focus on key time points in the data.
    - Train and validate the enhanced model, comparing its performance with the baseline.
6. **Evaluation and Comparison:**
    - Evaluate both models using the test set and compare their performance using MAE, MSE, R2.
    - Create a line chart to visualize the actual vs predicted stock prices for both models and analyze the differences in their predictive abilities.


## Requirements
- Python 3.x
- Libraries:
  - `TensorFlow`
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `Matplotlib`
  - `Seaborn`
 
## Results
- Base Model (Bidirectional LSTM): The initial model is trained and evaluated, and the results are compared to the enhanced model.
- Enhanced Model (LSTM with Sequential Self-Attention): The enhanced model demonstrates improved performance by focusing on relevant time points in the time series data.
- Comparison of Predictions: A line chart is generated comparing the actual stock prices with predictions from both models, showing the performance differences.
  
## Conclusion
This project demonstrates how deep learning techniques, particularly Bidirectional LSTMs and Sequential Self-Attention, can be applied to time series data for stock price prediction. The addition of a self-attention mechanism to the LSTM architecture improves the model’s ability to focus on critical time periods, resulting in better predictions of stock prices.
