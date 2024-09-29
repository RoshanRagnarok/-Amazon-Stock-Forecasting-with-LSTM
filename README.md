# LSTM Stock Price Prediction

## Overview

This project implements an LSTM model using PyTorch to predict the closing prices of Amazon (AMZN) stock based on historical data.

## Features

- Preprocesses stock data with sliding windows.
- Normalizes data using `MinMaxScaler`.
- Trains a single-layer LSTM model to predict future prices.
- Evaluates the model and visualizes the results with matplotlib.

## Setup

1. Install dependencies:
    ```bash
   pip install pandas numpy matplotlib torch scikit-learn
2.Place AMZN.csv in the project folder (contains Date and Close columns).<br>
3.Run the script:
```bash
     python stock_lstm.py




