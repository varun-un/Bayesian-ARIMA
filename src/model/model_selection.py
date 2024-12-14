import pmdarima
import pandas as pd
import os

def determine_arima_order(series, max_p=5, max_d=2, max_q=5, seasonal=True, m=1):
    """
    Determines the optimal ARIMA order (p, d, q) for a stock's data

    Returns a tuple of p, d, q ARIMA order for this ticker
    """

    # pmdarima call
    model = pmdarima.auto_arima(
        series,
        start_p=1,
        start_q=1,
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        seasonal=seasonal,
        m=m,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False,
        maxiter=100,
    )
    return model.order

if __name__ == "__main__":

    from ..preprocessing.preprocessor import load_data

    ticker = 'AAPL'
    processed_path = f"data/processed/{ticker}_processed.csv"
    df = pd.read_csv(processed_path, index_col='Date', parse_dates=True)
    log_returns = df['Log_Returns']
    
    order = determine_arima_order(log_returns)
    print(f"Optimal ARIMA order for {ticker}: {order}")
