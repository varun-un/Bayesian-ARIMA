from bayesian_arima import BayesianARIMA
from model_selection import determine_arima_order
import pandas as pd
from typing import Dict

class HierarchicalModel:
    """
    This class represents one ticker, which will have sector and industry data as well as 4 ARIMA models.
    Each model has different timeframes and seasonality. The range of data available with the yfinance api is 
    also different for each interval.
    - Monthly: Interval = 30 days, Seasonality = 12
    - Daily: Interval = 1 day, Seasonality = 5, Range = 20 years
        * https://www.investopedia.com/terms/w/weekendeffect.asp#:~:text=Key%20Takeaways,of%20the%20immediately%20preceding%20Friday.
        * https://www.researchgate.net/publication/225399137_The_day_of_the_week_effect_on_stock_market_volatility
    - Hourly: Interval = 1 hour, Seasonality = 6, Range = 2 years
    - Minute: Interval = 1 minute, Range = 1 month

    """


    def __init__(self, sector: str, industry: str):
        """
        Initialize with sector and industry names.
        """
        self.sector = sector
        self.industry = industry
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
