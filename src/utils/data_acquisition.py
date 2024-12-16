
import yfinance as yf
import pandas as pd
import os

def fetch_all_data(ticker, range, fetch_range, interval, end_date=None):
    """
    Fetch historical stock data for a given ticker.

    Parameters:
    - ticker (str): Stock ticker symbol (ex. 'AAPL')
    - range (int): Number of seconds to fetch data for
    - fetch_range (int): Max number of seconds to fetch in one request
    - interval (str): Data frequency ('1m', '1h', '1d', etc)
    - end_date (int): Unix timestamp for end date

    Returns:
    - pd.DataFrame
    """
    
    if end_date is None:
        end_date = int(pd.Timestamp.now().timestamp())

    # do we need to fetch data in chunks
    if fetch_range is not None:

        earliest_time = end_date - range
        fetch_start = earliest_time

        # fetch data in chunks
        while fetch_start < end_date:
            fetch_end = min(fetch_start + fetch_range, end_date)
            data_chunk = yf.download(ticker, interval=interval, start=fetch_start, end=fetch_end)

            # append chunks together to make one dataframe
            if fetch_start == earliest_time:
                data = data_chunk
            else:
                data = pd.concat([data, data_chunk])

    else:
        data = yf.download(ticker, interval=interval, period='max')

    return data


def save_stock_data(ticker, start_date, end_date, interval, save_path):
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
        save_stock_data(ticker, start_date, end_date, '1d', path)
