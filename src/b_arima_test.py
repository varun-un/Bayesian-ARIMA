# example_bayesian_arima.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import BayesianARIMA, determine_arima_order, adf_test

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

# use augmented Dickey-Fuller test to check for stationarity
stationary = adf_test(y, verbose=True)


# optimal ARIMA order
if not stationary:
    order = determine_arima_order(y, max_p=10, max_d=10, max_q=10, m=1)
else:
    order = determine_arima_order(y, max_p=10, max_d=1, max_q=10, m=1)
print(f"Optimal ARIMA order for {ticker}: {order}")


# initialize and train the Bayesian ARIMA model
p, d, q = order
bayesian_arima = BayesianARIMA(name="AAPL", p=p, d=d, q=q, seasonal=False, m=1)

# train the model
bayesian_arima.train(y=y, draws=10, tune=10, target_accept=0.75)

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
