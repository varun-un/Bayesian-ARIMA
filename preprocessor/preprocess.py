import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

"""
For the interval, I'm just gonna use closing times for the sake of simplicity
Gives 1 day period - using openings too could give roughly 16 hr and 8 hr off-cyclic periods, but after-market trading has different trends
    this would need differential indicators and 2 models trained: 1 for close -> open, 1 for open -> close
    trends due to after market are also less likely indicated by price action, but instead earnings calls, news, etc, so the model would struggle anyways
"""

def load_data(file_path):
    """
    Loads stock data from a CSV file.
    """
    df = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    return df['Close']

def calculate_log_returns(series):
    """
    Takes input of a pandas series of prices
    Returns a pandas series of log returns

    Log returns allow for easier comparison of returns over time
    """
    return np.log(series / series.shift(1)).dropna()

def check_stationarity(series, significance_level=0.05):
    """
    Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series)
    p_value = result[1]
    return p_value < significance_level

def difference_series(series, order=1):
    """
    Applies differencing to make the series stationary.
    """
    return series.diff(order).dropna()

def preprocess_data(raw_data_path, processed_data_path, differencing_order=1, continuous_diff=False):
    """
    Full preprocessing pipeline: load, calculate log returns, handle missing values, ensure stationarity.

    Parameters:
    - raw_data_path (str): Path to raw CSV data
    - processed_data_path (str): Path to save processed data
    - differencing_order (int): Order of differencing to apply
    - continuous_diff (bool): If True, apply differencing until series is stationary

    Returns:
    - pd.Series
    """
    series = load_data(raw_data_path)
    
    # Hfix missing values
    series = series.ffill().dropna()
    
    log_returns = calculate_log_returns(series)
    
    # Ensure stationarity
    if continuous_diff:
        max_diffs = 10      # randomly chosen upper bound escape case
        differencing_order = 0
        while not check_stationarity(log_returns) and differencing_order < max_diffs:
            differencing_order += 1
            log_returns = difference_series(log_returns)

        if differencing_order == max_diffs:
            raise ValueError("Series is still non-stationary after max of 10 differencing.")
        
    elif not check_stationarity(log_returns):
        log_returns = difference_series(log_returns, order=differencing_order)
        if not check_stationarity(log_returns):
            raise ValueError("Series is still non-stationary after differencing.")
        
    # else: series is already stationary
    
    # save data to file - will change for more streamlined data handling
    if not os.path.exists(os.path.dirname(processed_data_path)):
        os.makedirs(os.path.dirname(processed_data_path))
    log_returns.to_csv(processed_data_path, index=True, header=['Log_Returns'])
    print(f"Processed data saved to {processed_data_path}")
    
    return log_returns

if __name__ == "__main__":

    tickers = ['AAPL', 'MSFT', 'GOOG']
    for ticker in tickers:
        raw_path = f"data/raw/{ticker}.csv"
        processed_path = f"data/processed/{ticker}_processed.csv"
        try:
            preprocess_data(raw_path, processed_path)
        except ValueError as e:
            print(f"Preprocessing failed for {ticker}: {e}")