from bayesian_arima import BayesianARIMA
from model_selection import determine_arima_order
import pandas as pd
from typing import Dict

class HierarchicalModel:
    """
    This class represents one ticker, which will have 4 ARIMA models.
    Each model has different timeframes and seasonality. The range of data available with the yfinance api is 
    also different for each interval.
    - Monthly: Interval = 30 days, Seasonality = 12
    - Daily: Interval = 1 day, Seasonality = 5, Range = 20 years
        * https://www.investopedia.com/terms/w/weekendeffect.asp#:~:text=Key%20Takeaways,of%20the%20immediately%20preceding%20Friday.
        * https://www.researchgate.net/publication/225399137_The_day_of_the_week_effect_on_stock_market_volatility
    - Hourly: Interval = 1 hour, Seasonality = 6, Range = 2 years
    - Minute: Interval = 1 minute, Range = 1 month

    """


    def __init__(self):
        """
        Initializes the HierarchicalModel with empty models and seasonality values.
        """
        # dict of BayesianARIMA models for each timeframe
        self.models = {
            'monthly': None,
            'daily': None,
            'hourly': None,
            'minute': None
        }
        self.seasonality = {
            'monthly': 12,
            'daily': 5,
            'hourly': 6,
            'minute': 1
        }
        self.interval = {
            'monthly': '1mo',
            'daily': '1d',
            'hourly': '1h',
            'minute': '1m'
        }
        self.range = {
            'monthly': '20y',
            'daily': '20y',
            'hourly': '2y',
            'minute': '1mo'
        }


    def add_model(self, timeframe: str, model: BayesianARIMA):
        """
        Add a Bayesian ARIMA model for a specific timeframe. (usually an internal method)
        """
        if timeframe not in self.models:
            raise ValueError("Invalid timeframe. Choose from 'monthly', 'daily', 'hourly', 'minute'.")
        self.models[timeframe] = model

    def train_all(self, data: Dict[str, pd.Series], exog: Dict[str, pd.Series], sampler):
        """
        Train all ARIMA models using the provided data and sampler. The exogenous variables are optional.
        This will also create and add models for all time intervals.

        Parameters:
        - data: Dictionary of timeframes to series. The keys should be 'monthly', 'daily', 'hourly', 
                'minute', and the values should be pandas Series.
        - exog: Dictionary of timeframes to exogenous variables. The keys should be 'monthly', 'daily', 
                'hourly', 'minute', and the values should be pandas Series. If no exogenous variables, 
                set to `None`.
        - sampler: A sampler object to provide training data
        """
        for timeframe, series in data.items():
            sampled_series = sampler.sample(series)
            if exog is None:
                sampled_exog = None
            else:
                sampled_exog = exog.get(timeframe, None)

            # get seasonality and ARIMA order
            seasonality = self.seasonality[timeframe]
            p, d, q = determine_arima_order(sampled_series, seasonal=(seasonality > 1), m=seasonality)
            
            # create and train the models
            model = BayesianARIMA(p, d, q)
            model.train(sampled_series, sampled_exog)
            self.add_model(timeframe, model)

    def predict_all(self, steps: int) -> Dict[str, pd.Series]:
        """
        Generate forecasts from all models. Forward passes through the ARIMA models

        Parameters:
        - steps: Number of steps to forecast into the future.

        Returns:
        - dict: Dictionary of timeframes to forecasted series. Keys are 'monthly', 'daily', 'hourly', 'minute'.
                Values are pandas series of forecasted values and length `steps`.
        """
        forecasts = {}
        for timeframe, model in self.models.items():
            forecasts[timeframe] = model.predict(steps)
        return forecasts
    
    def predict_to_time(self, delta_t: pd.Timedelta) -> Dict[str, float]:
        """
        Generate forecasts from all models up to a specific time in the future.

        Parameters:
        - delta_t: The time into the future to forecast to.

        Returns:
        - dict: Dictionary of values predicted at the specific time in the future. Keys are 'monthly', 'daily', 'hourly', 'minute'.
                Values are the value at the specific time in the future. Linear interpolation is used for delta_t not divisible by the interval.
        """
        # number of steps to take based on interval
        delta_t_minutes = delta_t.total_seconds() / 60
        delta_t_hours = delta_t.total_seconds() / 3600
        delta_t_days = delta_t.total_seconds() / (3600 * 24)
        delta_t_months = delta_t.days / 30

        deltas = {
            'monthly': delta_t_months,
            'daily': delta_t_days,
            'hourly': delta_t_hours,
            'minute': delta_t_minutes
        }

        steps = {}      # number of steps to take for each interval
        for timeframe, delta in deltas.items():
            if delta % 1 != 0:
                # round up if not divisible by interval
                steps[timeframe] = int(delta) + 1  
            else:
                steps[timeframe] = int(delta)

        forecasts = {}      # forecasted values for each interval - each entry is a pd.Series
        for timeframe, model in self.models.items():
            forecasts[timeframe] = model.predict(steps[timeframe])

        # get the actual prediction value - take last value or interpolate
        predictions = {}

        for timeframe, forecast in forecasts.items():
            # if the key's delta is a decimal, interpolate by that decimal value
            if deltas[timeframe] % 1 != 0:
                # get the two closest values
                lower = forecast.iloc[-2]
                upper = forecast.iloc[-1]
                # interpolate
                predictions[timeframe] = lower + (upper - lower) * (deltas[timeframe] % 1)
            else:
                predictions[timeframe] = forecast.iloc[-1]

        return predictions
    
    def predict_to_time_labelled(self, delta_t: pd.Timedelta) -> Dict[str, pd.Series]:
        """
        Generate forecasts from all models up to a specific time in the future. Returns a dictionary of time-labelled forecasts.

        Similar to predict_to_time, but its return is a series of data points with time labels. Useful for plotting.

        Parameters:
        - delta_t: The time into the future to forecast to.

        Returns:
        - dict: Dictionary of time-labelled forecasts. Keys are 'monthly', 'daily', 'hourly', 'minute'.
                Values are pandas series of forecasted values with time labels.
        """
        # number of steps to take based on interval
        delta_t_minutes = delta_t.total_seconds() / 60
        delta_t_hours = delta_t.total_seconds() / 3600
        delta_t_days = delta_t.total_seconds() / (3600 * 24)
        delta_t_months = delta_t.days / 30

        deltas = {
            'monthly': delta_t_months,
            'daily': delta_t_days,
            'hourly': delta_t_hours,
            'minute': delta_t_minutes
        }

        steps = {}      # number of steps to take for each interval
        for timeframe, delta in deltas.items():
            if delta % 1 != 0:
                # round up if not divisible by interval
                steps[timeframe] = int(delta) + 1  
            else:
                steps[timeframe] = int(delta)

        forecasts = {}      # forecasted values for each interval - each entry is a pd.Series
        for timeframe, model in self.models.items():
            forecasts[timeframe] = model.predict(steps[timeframe])

        # give each forecast a time label
        labelled_forecasts = {}
        for timeframe, forecast in forecasts.items():
            # get the last time in the series
            last_time = forecast.index[-1]
            # create a new date range from the last time to the future time
            future_range = pd.date_range(start=last_time, periods=steps[timeframe], freq=model.interval)
            # create a new series with the future range
            labelled_forecasts[timeframe] = pd.Series(forecast.values, index=future_range)

            # if the key's delta is a decimal, interpolate by that decimal value. change the last value and its time
            if deltas[timeframe] % 1 != 0:
                # get the two closest values
                lower = labelled_forecasts[timeframe].iloc[-2]
                upper = labelled_forecasts[timeframe].iloc[-1]
                # interpolate
                interpolated_value = lower + (upper - lower) * (deltas[timeframe] % 1)
                # change the last value and its time
                labelled_forecasts[timeframe].iloc[-1] = interpolated_value
                labelled_forecasts[timeframe].index = labelled_forecasts[timeframe].index.shift(1)

        return labelled_forecasts


        