from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(endog=[50, 56, 59, 60, 56, 52, 59, 63, 64], 
              order=(2, 1, 1), 
              enforce_stationarity=True, 
              enforce_invertibility=True)

result = model.fit()

print(result.summary())