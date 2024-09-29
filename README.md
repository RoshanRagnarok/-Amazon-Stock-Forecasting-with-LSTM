LSTM Stock Price Prediction - README
Project Overview

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) network. We train the LSTM model on Amazon (AMZN) historical stock price data to predict the next day's closing price. The project covers the entire workflow, from data preprocessing to building, training, and testing the LSTM model using PyTorch.
Features

    Data Preprocessing:
        Load the stock price dataset.
        Transform the data for time series forecasting using a sliding window approach.
        Scale the data for better neural network performance.

    LSTM Model:
        Build an LSTM-based neural network using PyTorch.
        Train the network to minimize the mean squared error (MSE) loss on training data.
        Evaluate the model on both training and test data.

    Result Visualization:
        Plot actual vs. predicted stock prices for both training and test sets.

Project Structure

bash

|-- AMZN.csv                 # Dataset containing historical AMZN stock prices.
|-- stock_lstm.py            # Python script containing the entire workflow.
|-- README.md                # Project documentation.

Setup Instructions
Requirements

Make sure you have the following libraries installed:

bash

pip install pandas numpy matplotlib torch scikit-learn

Files

    AMZN.csv: This file contains historical Amazon stock prices. The relevant columns used in the project are Date and Close.
    stock_lstm.py: This is the main script that implements the LSTM model.

Data Preprocessing

    Load Data:
        The dataset (AMZN.csv) is loaded and the Date and Close columns are selected.

    Sliding Window Transformation:
        A sliding window of n_steps (in this case 7) is applied to shift the closing prices into past sequences to train the model to predict future values.

    Data Scaling:
        The data is scaled to the range [-1, 1] using the MinMaxScaler to standardize the values and improve model training.

Model Architecture

The LSTM model consists of:

    Input Layer: Receives a sequence of 7 historical prices.
    LSTM Layer: 1 LSTM layer with 4 hidden units.
    Fully Connected (FC) Layer: Outputs a single predicted closing price.

The model is trained using the Adam optimizer and MSE loss function.
Training

    The data is split into training (95%) and testing (5%) sets.
    Data is batched using PyTorch DataLoader for efficient training.
    During training, the model predicts stock prices and calculates the MSE loss. We optimize the model using backpropagation.

Testing

The model is validated on unseen data (the test set), and predictions are compared with actual stock prices.
Visualizing Results

The results are visualized by plotting the actual vs. predicted closing prices for both the training and test datasets.
Example Plots

    Training Data Plot:

    Test Data Plot:

Usage

To run the project:

    Clone the repository.
    Run the Python script (stock_lstm.py) to train the model and visualize the results.

bash

python stock_lstm.py

Hyperparameters

    Learning Rate: 0.001
    Epochs: 10
    Batch Size: 16
    Lookback Window: 7 (predict based on past 7 days)
    Hidden Size: 4

Conclusion

This project demonstrates the application of LSTM networks for stock price prediction. It highlights the importance of data preprocessing, including sliding window transformations and scaling, and provides a framework for building and evaluating LSTM models on financial time series data.
