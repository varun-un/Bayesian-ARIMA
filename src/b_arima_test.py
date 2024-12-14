# example_bayesian_arima.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from model import BayesianARIMA, determine_arima_order

# historical data for Apple Inc.
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-12-31'
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

# target variable
y = data['Close']

# handle missing values
y = y.dropna()

# plot the original series
plt.figure(figsize=(12, 6))
plt.plot(y, label='Adjusted Close Price')
plt.title(f'{ticker} Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# stationarity tests using the Augmented Dickey-Fuller test
def test_stationarity(timeseries):
    """
    Performs the Augmented Dickey-Fuller test to check stationarity.
    
    Parameters:
    - timeseries (pd.Series): The time series data.
    
    Returns:
    - None
    """
    print('Results of Augmented Dickey-Fuller Test:')
    result = adfuller(timeseries)
    labels = ['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result[:4], labels):
        print(f'{label}: {value}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    if result[1] < 0.05:
        print("Conclusion: The series is stationary.")
    else:
        print("Conclusion: The series is non-stationary.")

# Apply stationarity test
test_stationarity(y)

# optimal ARIMA order
order = determine_arima_order(y, max_p=20, max_d=20, max_q=20, seasonal=False, m=1)
print(f"Optimal ARIMA order for {ticker}: {order}")


# initialize and train the Bayesian ARIMA model
p, d, q = order
bayesian_arima = BayesianARIMA(name="AAPL", p=p, d=d, q=q, seasonal=False, m=1)

# train the model
bayesian_arima.train(y=y, draws=100, tune=100, target_accept=0.75)

try:
    bayesian_arima.save()
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# prepare for forecasting
# differenced target series
y_diff = y.diff(d).dropna().values

# get the last 'p' observations from the differenced series
last_observations = y_diff[-p:]

steps = 5  # forecasting the next 5 days

# Generate forecasts
forecasts_diff = bayesian_arima.predict(steps=steps, last_observations=last_observations)
print("Forecasted Differenced Values:")
print(forecasts_diff)

# Convert differenced forecasts back to original scale
last_actual = y.iloc[-1]
forecast_values = []
current_value = last_actual

for diff in forecasts_diff:
    current_value += diff
    forecast_values.append(current_value)

# forecast dates
last_date = y.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')  # 'B' for business days

forecast_series = pd.Series(forecast_values, index=forecast_dates, name='Forecast')

print("Forecasted Adjusted Close Prices:")
print(forecast_series)

# plot the forecasts alongside historical data
plt.figure(figsize=(12, 6))
plt.plot(y, label='Historical')
plt.plot(forecast_series, label='Forecast', marker='o')
plt.title(f'{ticker} Adjusted Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
