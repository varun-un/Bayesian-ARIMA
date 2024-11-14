
import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date, end_date, interval, save_path):
    """
    ticker (str): Stock ticker symbol (e.g., 'AAPL')
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    interval (str): Data frequency ('1m', '1h', '1d', etc)
    save_path (str): Path to save the fetched data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)

    # create and save to folder
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)
    print(f"Data for {ticker} saved to {save_path}")

if __name__ == "__main__":

    tickers = ['AAPL', 'MSFT', 'GOOG']

    # 'YYYY-MM-DD' format
    start_date = '2024-10-15'
    end_date = '2024-10-16'

    # get and save data for each ticker
    for ticker in tickers:
        path = f"data/raw/{ticker}.csv"
        fetch_stock_data(ticker, start_date, end_date, '1d', path)
