from .bayesian_sarima import BayesianSARIMA
from .model_selection import determine_sarima_order
from src.utils import invert_differencing, fetch_all_data, TradingTimeDelta
import yfinance as yf
import pandas as pd
from typing import Dict, Tuple
import pickle
import dill
from pathlib import Path
from src.ensemble import Ensemble
import numpy as np
from datetime import timedelta

class HierarchicalModel:
    """
    This class represents one ticker, which will have 3 ARIMA models.
    Each model has different timeframes and seasonality. The range of data available with the yfinance api is 
    also different for each interval.
    - Daily: Interval = 1 day, Seasonality = 5, Range = 20 years
        * https://www.investopedia.com/terms/w/weekendeffect.asp#:~:text=Key%20Takeaways,of%20the%20immediately%20preceding%20Friday.
        * https://www.researchgate.net/publication/225399137_The_day_of_the_week_effect_on_stock_market_volatility
    - Hourly: Interval = 1 hour, Seasonality = 6, Range = 2 years
    - Minute: Interval = 1 minute, Range = 1 month

    """


    def __init__(self, ticker: str, ensemble: Ensemble = None, memory_save: bool = False):
        """
        Initializes the HierarchicalModel with empty models and seasonality values.

        Parameters:
        - ticker: str: The ticker symbol of the stock.
        - ensemble: Ensemble: The ensemble method to use for combining the forecasts. Default is None.
        - memory_save: bool: Whether to save the models in memory for use after training. Default is False. If True,
                            after a model is trained, it will be saved to a file and removed from RAM.
        """

        self.ticker = ticker
        self.ensemble = ensemble
        self.memory_save = memory_save
        # dict of BayesianSARIMA models for each timeframe
        self.models = {
            'daily': None,
            'hourly': None,
            'minute': None
        }
        self.pickled_models = {
            'daily': None,
            'hourly': None,
            'minute': None
        }
        self.seasonality = {
            'daily': 5,
            'hourly': 6,
            'minute': 1
        }
        self.interval = {
            'daily': '1d',
            'hourly': '1h',
            'minute': '1m'
        }
        # deliberately choose smaller than max range to avoid memory issues
        self.range = {              # in seconds
            'daily': 10 * 365 * 24 * 60 * 60,
            'hourly': int(1.8 * 365 * 24 * 60 * 60),
            'minute': 13 * 24 * 60 * 60
        }

        # how much data you can fetch in one GET request
        self.fetch_range = {        # in seconds
            'daily': None,
            'hourly': None,
            'minute': 7 * 24 * 60 * 60
        }

    def train_models(self, num_draws: int = 250, num_tune: int = 250, target_accept: float = 0.95):
        """
        Train the BayesianSARIMA models for each timeframe.

        Trains it by default on the max range of data available for each timeframe.

        """
        
        for timeframe, seasonality in self.seasonality.items():

            # current time in unix seconds
            current_time = int(pd.Timestamp.now().timestamp())
            
            data = fetch_all_data(self.ticker, self.range[timeframe], self.fetch_range[timeframe], self.interval[timeframe], end_date=current_time)

            # target variable
            y = data['Close']

            # handle missing values
            y = y.dropna()

            print(f"Determining order for {timeframe} model...")

            # generate the models themselves
            order = determine_sarima_order(y, max_p=10, max_d=4, max_q=5, m=seasonality, max_P=2, max_D=2, max_Q=2)   
            # order = (5, 1, 1, 1, 1, 2)       # example order for testing
            p, d, q, P, D, Q = order

            self.models[timeframe] = BayesianSARIMA(name=f"{self.ticker}_{timeframe}", m=seasonality, p=p, d=d, q=q, P=P, D=D, Q=Q)

            print(f"Training {timeframe} model...")

            # train the model
            self.models[timeframe].train(y=y, draws=num_draws, tune=num_tune, target_accept=target_accept)

            # save the models
            try:
                self.pickled_models[timeframe] = self.models[timeframe].save()

                if self.memory_save:
                    self.models[timeframe] = None
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def predict_to_time(self, end_time: pd.Timestamp) -> Tuple[Dict[str, float], Dict[str, pd.Series]]:
        """
        Generate forecasts from all models up to a specific time in the future.

        Parameters:
        - end_time: pd.Timestamp: The time to predict to.

        Returns:
        - dict, dict: Dictionary of values predicted at the specific time in the future. Keys are 'daily', 'hourly', 'minute'.
                Values are the value at the specific time in the future. Linear interpolation is used for delta_t not divisible by the interval.
        """

        # get the delta time in seconds
        start_time = pd.Timestamp.now()
        delta_t = TradingTimeDelta(start_time=start_time, end_time=end_time)

        # number of steps to take based on interval
        delta_t_hours = delta_t.get_delta_hours()
        delta_t_days = delta_t.get_delta_days()
        delta_t_minutes = delta_t.get_delta_minutes()

        deltas = {
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

        forecasts = {}      # forecasted values for each interval - each entry is a np array
        for timeframe, model in self.models.items():

            data = fetch_all_data(self.ticker, self.range[timeframe], self.fetch_range[timeframe], self.interval[timeframe])
            y = data['Close']
            y = y.dropna()


            y_diff = y.diff(model.d).dropna().values
            last_observations = y_diff


            forecasts[timeframe] = model.predict(steps[timeframe], last_observations=last_observations)

            # fix the differncing
            forecasts[timeframe] = invert_differencing(forecasts[timeframe], model.d, y)

            # print(f"Forecasted {timeframe} values: {forecasts[timeframe]}")


        # get the actual prediction value - take last value or interpolate
        predictions = {}

        for timeframe, forecast in forecasts.items():
            try:
                # if the key's delta is a decimal, interpolate by that decimal value
                if deltas[timeframe] % 1 != 0:
                    # get the two closest values
                    lower = forecast[-2]
                    upper = forecast[-1]
                    # interpolate
                    predictions[timeframe] = lower + (upper - lower) * (deltas[timeframe] % 1)
                else:
                    predictions[timeframe] = forecast[-1]
            except Exception as e:
                # not enough data to interpolate
                predictions[timeframe] = forecast[-1]

        # label the forecasts with time labels
        labelled_forecasts = {}
        for timeframe, forecast in forecasts.items():
            # get the last time in the series
            last_time = TradingTimeDelta.get_next_trading_time(pd.Timestamp.now())

            # create a new date range from the last time to the future time
            future_range = pd.date_range(start=last_time, periods=steps[timeframe], freq=self.interval[timeframe])

            # if minute or hour model, we need custom date range
            if timeframe == 'minute':
                future_range = TradingTimeDelta.generate_trading_timestamps(last_time, steps[timeframe], increment=timedelta(minutes=1))
            elif timeframe == 'hourly':
                future_range = TradingTimeDelta.generate_trading_timestamps(last_time, steps[timeframe], increment=timedelta(hours=1))

            # create a new series with the future range
            labelled_forecasts[timeframe] = pd.Series(forecast, index=future_range)

        return predictions, labelled_forecasts
    
    def predict_value(self, end_time: pd.Timestamp) -> float:
        """
        Generate a forecasted value at a specific time in the future.
        Uses the ensemble method if it is provided.
        
        Parameters:
        - end_time: pd.Timestamp: The time to predict to.
        
        Returns:
        - float: The forecasted value at the specific time in the future.
        """
        delta_t = TradingTimeDelta(start_time=pd.Timestamp.now(), end_time=end_time)
        secs = delta_t.get_delta_seconds()          # use as exog variable for ensemble

        predictions, _ = self.predict_to_time(end_time)
        
        if self.ensemble is not None:
            
            # turn the predictions into a np vector
            preds = [predictions['daily'], predictions['hourly'], predictions['minute']]
            preds = np.array(preds)

            # get the ensemble prediction
            ensemble_pred = self.ensemble.ensemble(forecasts=preds, exog=secs)
            return ensemble_pred
        else:
            return sum(predictions.values()) / len(predictions)
    
    def save(self):
        """
        Save the hierarchical model to a file.
        """
        filename = Path(__file__).parent.parent.parent / f"models/hierarchical/{self.ticker}.pkl"

        # save everything to a pickle file except the models
        to_save = {
            'ticker': self.ticker,
            'seasonality': self.seasonality,
            'interval': self.interval,
            'range': self.range,
            'fetch_range': self.fetch_range,
            'pickled_models': self.pickled_models,
            'ensemble': self.ensemble
        }


        with open(filename, 'wb') as f:
            dill.dump(to_save, f)

    def load(self, filename: str = None):
        """
        Load the hierarchical model from a file.
        """
            
        if filename is None:
            filename = Path(__file__).parent.parent.parent / f"models/hierarchical/{self.ticker}.pkl"

        with open(filename, 'rb') as f:
            loaded = dill.load(f)

            self.ticker = loaded['ticker']
            self.seasonality = loaded['seasonality']
            self.interval = loaded['interval']
            self.range = loaded['range']
            self.fetch_range = loaded['fetch_range']
            self.pickled_models = loaded['pickled_models']
            self.ensemble = loaded['ensemble']

        # load the models
        for timeframe, model_path in self.pickled_models.items():
            if model_path is not None:

                # naming convention is {ticker}_{timeframe}-{p}-{d}-{q}-{P}-{D}-{Q}.pkl
                # extract the order from the filename
                order = model_path.stem.split('_')[-1].split('-')[1:]

                # convert to integers
                order = list(map(int, order))

                print(f"Loading {timeframe} model with order: {order}")

                self.models[timeframe] = BayesianSARIMA(name=f"{self.ticker}_{timeframe}", m=self.seasonality[timeframe], p=order[0], d=order[1], q=order[2], P=order[3], D=order[4], Q=order[5])

                # load the model
                self.models[timeframe].load(model_path)