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
        stepwise=False,             # False => grid search, True => opt loop
        maxiter=200,
        max_order=None,
        information_criterion='aic',     # better at future predictions than 'bic'
    )
    p, d, q = model.order

    p = max(p, 1)       # p must be at least 1
    q = max(q, 1)       # q must be at least 1

    return p, d, q

def determine_sarima_order(series, max_p=5, max_d=2, max_q=5, m=1, max_P=2, max_D=1, max_Q=2):
    """
    Determines the optimal SARIMA order (p, d, q, P, D, Q) for a stock's data

    Same as determine_arima_order, but more options for seasonal data

    Returns a tuple of p, d, q, P, D, Q SARIMA order for this ticker
    """

    seasonality = (m != 1)

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
        start_P=1,
        start_Q=1,
        max_P=max_P,
        max_Q=max_Q,
        max_D=max_D,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=False,             # False => grid search, True => opt loop
        maxiter=300,                # more iterations for seasonal
        max_order=None,
        information_criterion='aic',     # better at future predictions than 'bic'
    )
    p, d, q = model.order
    P, D, Q, _ = model.seasonal_order

    p = max(p, 1)       # p must be at least 1
    q = max(q, 1)       # q must be at least 1

    # if we're doing seasonal, make sure P and Q are at least 1
    if seasonality:
        P = max(P, 1)
        Q = max(Q, 1)
    else:
        P, D, Q = 0, 0, 0

    return p, d, q, P, D, Q

if __name__ == "__main__":

    from src.utils import load_data

    ticker = 'AAPL'
    processed_path = f"data/processed/{ticker}_processed.csv"
    df = pd.read_csv(processed_path, index_col='Date', parse_dates=True)
    log_returns = df['Log_Returns']
    