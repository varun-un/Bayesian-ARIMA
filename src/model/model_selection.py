import pmdarima
import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

def adf_test(series, signif=0.05, verbose=False):
    """
    Perform ADF test on a series. Returns True if series is stationary, False otherwise.
    
    Parameters:
    - series (pd.Series): The series to test.
    - signif (float): The significance level. Default is 0.05.
    - verbose (bool): Whether to print results. Default is False.
    """

    result = adfuller(series, autolag='AIC')
    pval = result[1]

    if verbose:
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {pval}')
        print(f'Critical Values:')
        for key, value in result[4].items():
            print(f'   {key}: {value}')

    return pval <= signif


def determine_arima_order(series, max_p=5, max_d=2, max_q=5, m=1):
    """
    Determines the optimal ARIMA order (p, d, q) for a stock's data

    Returns a tuple of p, d, q ARIMA order for this ticker
    """

    seasonality = m != 1

    # pmdarima call
    model = pmdarima.auto_arima(
        series,
        start_p=5,
        start_q=5,
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        seasonal=seasonality,
        m=m,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        maxiter=200,
        max_order=6,
        information_criterion='aic',     # better at future predictions than 'bic'
    )
    p, d, q = model.order

    p = max(p, 1)       # p must be at least 1
    q = max(q, 1)       # q must be at least 1

    return p, d, q

if __name__ == "__main__":

    from ..utils.preprocessor import load_data

    ticker = 'AAPL'
    processed_path = f"data/processed/{ticker}_processed.csv"
    df = pd.read_csv(processed_path, index_col='Date', parse_dates=True)
    log_returns = df['Log_Returns']
    
    order = determine_arima_order(log_returns)
    print(f"Optimal ARIMA order for {ticker}: {order}")
