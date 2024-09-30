import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Sample stock price data
data = {
    'Time': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'Price': [50, 56, 59, 60, 56, 52, 59, 63, 64]
}

df = pd.DataFrame(data)

# set 'Time' as index to treat it as a time series
df.set_index('Time', inplace=True)

# plot raw data
plt.figure(figsize=(10, 5))
plt.plot(df, marker='o')
plt.title('Stock Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# 1st order differencing to remove any trend
df_diff = df.diff().dropna()

plt.figure(figsize=(10, 5))
plt.plot(df_diff, marker='o', color='green')
plt.title('Differenced Stock Prices (First Order)')
plt.xlabel('Time')
plt.ylabel('Differenced Price')
plt.show()


# ACF and PACF for the differenced data
plt.figure(figsize=(12, 5))

plt.subplot(121)
plot_acf(df_diff, ax=plt.gca(), lags=7)
plt.title('ACF of Differenced Data')

plt.subplot(122)
plot_pacf(df_diff, ax=plt.gca(), lags=4)
plt.title('PACF of Differenced Data')

plt.tight_layout()
plt.show()

#------------------------------------ ACF Analysis

from statsmodels.tsa.arima.model import ARIMA


# Step 4: Fit ARIMA(1,1,1) model
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Step 5: Predict future prices for time steps 9 and 10
forecast = model_fit.forecast(steps=2)
print(f"Predicted prices for timesteps 9 and 10: {forecast.values}")

