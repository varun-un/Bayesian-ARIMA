# Bayesian code inspired off math described in: https://www.columbia.edu/~mh2078/MachineLearningORFE/MCMC_Bayes.pdf

import pymc as pm
import numpy as np
import pandas as pd
from typing import Optional
import aesara.tensor as tt

import pickle
from typing import Tuple
from pathlib import Path



class BayesianARIMA:
    def __init__(self, name: str, p: int, d: int, q: int, seasonal: bool = False, m: int = 1):
        """
        Initialize Bayesian ARIMA with specified order.
        
        Parameters:
        - name: Name of the model. Will be used for saving and loading.
        - p: Number of autoregressive terms.
        - d: Number of nonseasonal differences.
        - q: Number of moving average terms.
        - seasonal: Whether to include seasonal components.
        - m: Seasonal period.
        """
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.trace = None

    def train(self, y: pd.Series):
        """
        Train the Bayesian ARIMA model using PyMC3.
        
        Samples from the posterior distribution using MCMC.
        Builds Bayesian model around each AR and MA parameter
        """
        # Differencing
        y_diff = y.diff(self.d).dropna().values

        self.model = pm.Model()

        with self.model:
            # priors for AR coefficients
            phi = pm.Normal('phi', mu=0, sigma=10, shape=self.p)
            
            # priors for MA coefficients
            theta = pm.Normal('theta', mu=0, sigma=10, shape=self.q)
            
            # prior for the noise - half normal to keep positive
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Initialize mu to zeros with shape (N-p,)
            mu = tt.zeros(len(y_diff) - self.p)
            
            # Use summations to sum up AR, MA, and error terms (and exog if exists)

            # AR component
            for i in range(1, self.p + 1):
                mu += phi[i - 1] * y_diff[self.p - i : -i] 
            
            # MA component
            if self.q > 0:
                #  latent error terms
                eps = pm.Normal('eps', mu=0, sigma=sigma, shape=len(y_diff) + self.q)   # pad shape (N+q,) to avoid indexing errors
                
                # MA terms
                for j in range(1, self.q + 1):
                    mu += theta[j - 1] * eps[self.p + self.q - j : -j]      # shift by p+q to match AR terms - indexing due to q-padding
            
            # likelihood of observations on the differenced series - Bayesian update step
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_diff[self.p:])
            
            # Sampling - PyMC3 uses Hamiltonian Monte Carlo (MCMC) for sampling
            self.trace = pm.sample(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True)

    def predict(self, steps: int, last_observations: Optional[np.ndarray] = None) -> pd.Series:
        """
        Generate forecasts using the posterior samples.
        
        Parameters:
        - steps: Number of future steps to predict.
        - last_observations: Last p observations from the differenced series.
        
        Returns:
        - pd.Series: Forecasted values.
        """
        if self.trace is None:
            raise ValueError("Model has not been trained yet.")
        
        # posterior means for AR, MA, and sigma
        # Bayesian posterior distribution
        phi_post = self.trace.posterior['phi'].mean(dim=['chain', 'draw']).values
        theta_post = self.trace.posterior['theta'].mean(dim=['chain', 'draw']).values
        sigma_post = self.trace.posterior['sigma'].mean(dim=['chain', 'draw']).values
        
        # forward forecasts
        forecast = []
        
        # Get the last p observations for use in time series forecasting
        if last_observations is None:
            raise ValueError("Observations must be provided for forecasting")
        
        # if last_observations is not the right length, take the last observation and repeat it to match p
        if len(last_observations) != self.p:
            last_observations = np.repeat(last_observations[-1], self.p)
        
        # get the last p observations for AR components
        ar_terms = list(last_observations[-self.p:])

        # use posterior samples for eps if q > 0
        if self.q > 0:
            eps_samples = self.trace.posterior['eps'].values  # shape - (chains, draws, len(y_diff) + q)
            # use the error terms from the posterior samples, and calculate the mean to help match shapes
            eps_post_mean = eps_samples.mean(axis=(0, 1))  # shape - (len(y_diff) + q,)

            ma_terms = list(eps_post_mean[-self.q:])    # get the last q error terms
        else:
            ma_terms = []
        
        # since multi-dimensional, dot products can be used to do lienar combinations of weights with AR, MA terms
        for step in range(steps):

            ar_component = np.dot(phi_post, ar_terms[-self.p:])
            
            ma_component = np.dot(theta_post, ma_terms[-self.q:]) if self.q > 0 else 0
            
            # sample from the noise distribution
            epsilon = np.random.normal(0, sigma_post)
            
            # sum components for total arima forecast
            y_hat = ar_component + ma_component + epsilon
            forecast.append(y_hat)
            
            # update state
            ar_terms.append(y_hat)
            if self.q > 0:
                ma_terms.append(epsilon)
        
        # series data 
        forecast_series = pd.Series(forecast, name='Forecast')
        return forecast_series
    
    
    
    def save(self) -> str:
        """
        Save the model to a file.
        
        Returns:
        - str: Path to the saved model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        filename = Path(__file__).parent / f"../../models/arima/{self.name}.pkl"
        
        with open(filename, "wb") as f:
            pickle.dump({'model': self.model, 'trace': self.trace}, f)
        
        return filename
    
    def load(self, filename: str = None):
        """
        Load the model from a file.
        
        Parameters:
        - filename: Path to the saved model. If None, uses the default naming convention.
        """

        if filename is None:
            filename = Path(__file__).parent / f"../../models/arima/{self.name}.pkl"

        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.trace = data['trace']
