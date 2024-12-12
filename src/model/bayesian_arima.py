# Bayesian code inspired off math described in: https://www.columbia.edu/~mh2078/MachineLearningORFE/MCMC_Bayes.pdf

import pymc3 as pm
import numpy as np
import pandas as pd
from typing import Optional
import theano.tensor as tt
from model_persistence import save_model, load_model

class BayesianARIMA:
    def __init__(self, p: int, d: int, q: int, seasonal: bool = False, m: int = 1):
        """
        Initialize Bayesian ARIMA with specified order.
        
        Parameters:
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

    def train(self, y: pd.Series, exog: Optional[pd.Series] = None):
        """
        Train the Bayesian ARIMA model using PyMC3.
        
        Samples from the posterior distribution using MCMC.
        Builds Bayesian model around each AR and MA parameter
        """
        # Differencing
        y_diff = y.diff(self.d).dropna().values
        if exog is not None:
            exog_diff = exog.diff(self.d).dropna().values
            
            # exog needs to be made same size as primary series
            exog_diff = exog_diff[self.p:]
        else:
            exog_diff = None

        with pm.Model() as self.model:
            # Priors for AR coefficients
            phi = pm.Normal('phi', mu=0, sigma=10, shape=self.p)
            
            # Priors for MA coefficients
            theta = pm.Normal('theta', mu=0, sigma=10, shape=self.q)
            
            # if the exog is being used, add a prior for the beta coefficient
            if exog_diff is not None:
                if exog_diff.ndim > 1:
                    beta = pm.Normal('beta', mu=0, sigma=10, shape=exog_diff.shape[1])
                else:
                    beta = pm.Normal('beta', mu=0, sigma=10)
            
            # prior for the noise - half normal to keep positive
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # initialize mu to zeros
            mu = tt.zeros_like(y_diff)
            
            # Use summations to sum up AR, MA, and error terms (and exog if exists)

            # AR component
            for i in range(self.p):
                mu += phi[i] * y_diff[self.p - i - 1 : -i -1]
            
            # MA component
            if self.q > 0:
                # Initialize latent error terms
                eps = pm.Normal('eps', mu=0, sigma=sigma, shape=len(y_diff))
                
                # Incorporate MA terms
                for j in range(self.q):
                    if j < len(eps):
                        mu += theta[j] * eps[self.p + j - 1]
            
            # Exogenous variables
            if exog_diff is not None:
                if exog_diff.ndim > 1:
                    mu += tt.dot(beta, exog_diff.T)
                else:
                    mu += beta * exog_diff
            
            # likelihood of observations on the differenced series - Bayesian update step
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_diff[self.p:])
            
            # Sampling - PyMC3 uses Hamiltonian Monte Carlo (MCMC) for sampling
            self.trace = pm.sample(draws=1000, tune=1000, target_accept=0.95, return_inferencedata=True)

    def predict(self, steps: int, last_observations: Optional[np.ndarray] = None, exog_future: Optional[pd.Series] = None) -> pd.Series:
        """
        Generate forecasts using the posterior samples.
        
        Parameters:
        - steps: Number of future steps to predict.
        - last_observations: Last p observations from the differenced series.
        - exog_future: Future exogenous variables.
        
        Returns:
        - pd.Series: Forecasted values.
        """
        if self.trace is None:
            raise ValueError("Model has not been trained yet.")
        
        # Extract posterior means for AR, MA, and sigma
        # Bayesian posterior distribution
        phi_post = self.trace.posterior['phi'].mean(dim=['chain', 'draw']).values
        theta_post = self.trace.posterior['theta'].mean(dim=['chain', 'draw']).values
        sigma_post = self.trace.posterior['sigma'].mean(dim=['chain', 'draw']).values
        
        if exog_future is not None:
            exog_future_diff = exog_future.diff(self.d).dropna().values
            exog_future_diff = exog_future_diff[-steps:]
        
        # forward forecasts
        forecast = []
        
        # Get the last p observations for use in time series forecasting
        if last_observations is None:
            raise ValueError("Last observations must be provided for forecasting.")
        ar_terms = list(last_observations[-self.p:])
        ma_terms = [0] * self.q  # Initialize MA terms with zeros
        
        # Since multi-dimensional, dot products can be used to do lienar combinations of weights with AR, MA terms
        for step in range(steps):

            ar_component = np.dot(phi_post, ar_terms[-self.p:])
            
            ma_component = np.dot(theta_post, ma_terms[-self.q:]) if self.q > 0 else 0
            
            # exogenous component
            exog_component = 0
            if exog_future is not None:
                exog_component = np.dot(self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values, exog_future_diff[step])
            
            # sample from the noise distribution
            epsilon = np.random.normal(0, sigma_post)
            
            # sum components for total arima forecast
            y_hat = ar_component + ma_component + exog_component + epsilon
            forecast.append(y_hat)
            
            # Update state
            ar_terms.append(y_hat)
            if self.q > 0:
                ma_terms.append(epsilon)
        
        # series data 
        forecast_series = pd.Series(forecast, name='Forecast')
        return forecast_series
