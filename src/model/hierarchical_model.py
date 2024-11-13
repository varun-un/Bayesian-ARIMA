from bayesian_arima import BayesianARIMA
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

