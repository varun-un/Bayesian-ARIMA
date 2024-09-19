# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Stock price data you provided
data = {
    'Time': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'Price': [50, 56, 59, 60, 56, 52, 59, 63, 64]
}

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Set 'Time' as index to treat it as a time series
df.set_index('Time', inplace=True)

# Plot the raw data
plt.figure(figsize=(10, 5))
plt.plot(df, marker='o')
plt.title('Stock Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# First order differencing to remove any trend
df_diff = df.diff().dropna()

# Plot the differenced data
plt.figure(figsize=(10, 5))
plt.plot(df_diff, marker='o', color='green')
plt.title('Differenced Stock Prices (First Order)')
plt.xlabel('Time')
plt.ylabel('Differenced Price')
plt.show()

