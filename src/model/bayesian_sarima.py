import pymc as pm
import numpy as np
import pandas as pd
from typing import Optional
import pytensor.tensor as pt
import pickle
import dill
from typing import Tuple
from pathlib import Path


class BayesianSARIMA:
    def __init__(self, name: str, p: int, d: int, q: int, m: int = 1, P: int = 0, D: int = 0, Q: int = 0):
        """
        Initialize Bayesian SARIMA with specified order.
        
        Parameters:
        - name: Name of the model for saving/loading.
        - p, d, q: Non-seasonal ARIMA orders.
        - m: Seasonal period.
        - P, D, Q: Seasonal ARIMA orders.
        """
        self.name = f"{name}-{p}-{d}-{q}-{P}-{D}-{Q}"
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = (m > 1)
        self.m = m
        self.P = P
        self.D = D
        self.Q = Q

        self.model = None
        self.trace = None

    def seasonal_difference(self, y: np.ndarray, D: int, m: int) -> np.ndarray:
        """
        Apply seasonal differencing (1 - B^m)^D to the series y.
        
        Parameters:
        - y: Original time series (already differenced d times if needed)
        - D: Seasonal differencing order
        - m: Seasonal period

        Returns:
        - y_diff: Seasonally differenced series
        """
        y_diff = y.copy()
        for _ in range(D):
            # Seasonal differencing: y_t <- y_t - y_{t-m}
            y_diff = y_diff[m:] - y_diff[:-m]
        return y_diff

    def train(self, y: pd.Series, draws: int = 1000, tune: int = 1000, target_accept: float = 0.95):
        """
        Train the Bayesian ARIMA model using PyMC.
        
        Steps:
        1. Nonseasonal differencing d times.
        2. If seasonal=True and D>0, apply seasonal differencing D times at lag m.
        3. Define priors for AR, MA, and seasonal AR, MA coefficients.
        4. Construct mu from AR, MA, Seasonal AR, Seasonal MA terms.
        5. Define the likelihood and sample from the posterior.

        Parameters:
        - y: Time series data.
        - draws: Number of samples to draw (pymc sampler for MCMC).
        - tune: Number of tuning steps to tune the sampler.
        - target_accept: Target acceptance rate for the sampler.
        """
        # nonseasonal differencing d times
        y_ndiff = y.diff(self.d).dropna().values  # y.diff(d) applied d times
        
        # if seasonal and D>0, apply seasonal differencing
        if self.seasonal and self.D > 0 and self.m > 1:
            y_diff = self.seasonal_difference(y_ndiff, self.D, self.m)
        else:
            y_diff = y_ndiff

        # length after differencing
        N = len(y_diff)
        if N <= max(self.p, self.q, self.P*self.m, self.Q*self.m):
            raise ValueError("Not enough data after differencing for given ARIMA and seasonal orders")

        self.model = pm.Model()

        with self.model:
            # priors for AR coefficients
            phi = pm.Normal('phi', mu=0, sigma=10, shape=self.p)
            
            # priors for MA coefficients
            theta = pm.Normal('theta', mu=0, sigma=10, shape=self.q)

            if self.seasonal and self.m > 1:
                # seasonal AR coefficients
                PHI = pm.Normal('PHI', mu=0, sigma=10, shape=self.P)    # PHI_1, ..., PHI_P
                # seasonal MA coefficients
                THETA = pm.Normal('THETA', mu=0, sigma=10, shape=self.Q) # THETA_1, ..., THETA_Q
            else:
                PHI = None
                THETA = None

            # prior for the noise - half normal to keep positive
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Initialize mu
            # construct mu for observations from indices max(p, P*m) onward to ensure lag availability
            start_index = max(self.p, self.P*self.m if self.seasonal else 0)
            mu = pt.zeros(N - start_index)

            # Convert y_diff to tensor
            y_diff_tensor = pt.as_tensor_variable(y_diff)

            # AR component (nonseasonal)
            # for i in [1, p], mu_t += phi_i * y_{t-i}
            # Slicing: y_diff_tensor[start_index - i : -i] aligns AR terms with mu's shape
            ar_contributions = []
            for i in range(1, self.p + 1):
                # y_{t-i}: slice y_diff starting from start_index - i to -i
                ar_contributions.append(phi[i - 1] * y_diff_tensor[start_index - i : -i])
            if ar_contributions:
                ar_sum = pm.math.sum(ar_contributions, axis=0)
                mu += ar_sum

            # Seasonal AR component
            # If seasonal, for I in [1, P], mu_t += PHI_I * y_{t - I*m}
            if self.seasonal and self.P > 0 and self.m > 1:
                sar_contributions = []
                for I in range(1, self.P + 1):
                    # seasonal lag: I*m
                    sar_contributions.append(PHI[I - 1] * y_diff_tensor[start_index - I*self.m : -I*self.m])
                if sar_contributions:
                    sar_sum = pm.math.sum(sar_contributions, axis=0)
                    mu += sar_sum

            # MA terms - latent error (epsilons)
            # Extend the q padding seasonal Q*m to avoid indexing errors and for seasonal
            pad = self.q + (self.Q * self.m if self.seasonal else 0)
            eps = pm.Normal('eps', mu=0, sigma=sigma, shape=N + pad)

            # Nonseasonal MA component
            # for j in [1, q], mu_t += theta_j * eps_{t-j}
            ma_contributions = []
            for j in range(1, self.q + 1):
                ma_contributions.append(theta[j - 1] * eps[start_index + pad - j : -j])
            if ma_contributions:
                ma_sum = pm.math.sum(ma_contributions, axis=0)
                mu += ma_sum

            # seasonal MA component
            # for j in [1, Q], mu_t += THETA_j * eps_{t - j*m}
            if self.seasonal and self.Q > 0 and self.m > 1:
                sma_contributions = []
                for j in range(1, self.Q + 1):
                    sma_contributions.append(THETA[j - 1] * eps[start_index + pad - j*self.m : -j*self.m])
                if sma_contributions:
                    sma_sum = pm.math.sum(sma_contributions, axis=0)
                    mu += sma_sum

            # likelihood of observations on the differenced series - Bayesian update step
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_diff[start_index:])

            # Sampling - PyMC3 uses Hamiltonian Monte Carlo (MCMC) for sampling
            self.trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True)

    def predict(self, steps: int, last_observations: Optional[np.ndarray] = None, noise_scale: float = 0.1) -> pd.Series:
        """
        Generate forecasts using the posterior samples.

        To forecast:
        - Use the posterior means of phi, theta, PHI, THETA, and sigma.
        - Initialize AR, MA, and seasonal AR, MA states from last_observations and eps_post.
        - Iteratively forecast steps ahead.

        Parameters:
        - steps: Number of future steps to predict.
        - last_observations: Last max(p, P*m) observations from the differenced series.
        - noise_scale: Scale factor for the noise. Default is 0.1. A factor of 0 will make the model deterministic.

        Returns:
        - pd.Series: Forecasted values.
        """
        if self.trace is None:
            raise ValueError("Model has not been trained yet.")

        # same as arima - take from posterior distribution
        phi_post = self.trace.posterior['phi'].mean(dim=['chain', 'draw']).values
        theta_post = self.trace.posterior['theta'].mean(dim=['chain', 'draw']).values
        sigma_post = self.trace.posterior['sigma'].mean(dim=['chain', 'draw']).values

        if self.seasonal and self.P > 0:
            PHI_post = self.trace.posterior['PHI'].mean(dim=['chain', 'draw']).values
        else:
            PHI_post = None

        if self.seasonal and self.Q > 0:
            THETA_post = self.trace.posterior['THETA'].mean(dim=['chain', 'draw']).values
        else:
            THETA_post = None

        # need at least max(p, P*m) prev observations to forecast
        req_length = max(self.p, self.P * self.m if (self.seasonal and self.m > 1) else 0)
        if last_observations is None or len(last_observations) < req_length:
            raise ValueError(f"Must provide at least {req_length} observations for forecasting.")

        # gonna pad last obs to meet the required
        if len(last_observations) > req_length:
            last_observations = last_observations[-req_length:]

        # Extract posterior eps to initialize MA terms - if q or Q is nonzero
        q_total = self.q + (self.Q * self.m if self.seasonal else 0)
        if q_total > 0:
            eps_samples = self.trace.posterior['eps'].values
            eps_mean = eps_samples.mean(axis=(0, 1))

            # Initialize MA terms from last q_total eps
            ma_terms = list(eps_mean[-q_total:])
        else:
            ma_terms = []

        # Initialize AR and seasonal AR terms
        # For AR: we need p recent differenced values
        # For seasonal AR: we need P*m differenced values
        # last_observations now contains max(p, P*m) values

        # forecast the differenced series

        forecast = []

        # treat last_observations as AR/seasonal AR terms
        # nonseasonal AR terms:
        ar_terms = list(last_observations[-self.p:]) if self.p > 0 else []

        # seasonal AR terms:
        if self.seasonal and self.P > 0 and self.m > 1:
            # extract seasonal AR terms
            sar_terms = list(last_observations[-self.P*self.m:])
        else:
            sar_terms = []

        for step in range(steps):
            # compute AR component
            ar_component = 0
            if self.p > 0:
                ar_component = np.dot(phi_post, ar_terms[-self.p:])

            # seasonal AR component
            sar_component = 0
            if self.seasonal and self.P > 0 and self.m > 1:

                # for each seasonal AR coefficient PHI_post[i], multiply by y_{t - (i+1)*m}

                # sar_terms stores last P*m observations
                # for each I in [1,P], we take sar_terms[-I*m]
                for I in range(1, self.P + 1):
                    sar_component += PHI_post[I - 1] * sar_terms[-I*self.m]

            # MA component
            ma_component = 0
            if self.q > 0:
                ma_component += np.dot(theta_post, ma_terms[-self.q:])

            # seasonal MA component
            if self.seasonal and self.Q > 0 and self.m > 1:

                # similar logic to AR above
                for J in range(1, self.Q + 1):
                    ma_component += THETA_post[J - 1] * ma_terms[-J*self.m]

            # sample noise - may change in the future to be deterministic but models random sampling of Bayesian model anyways
            epsilon = np.random.normal(0, sigma_post) * noise_scale         # scale down the noise, otherwise it tends to dominate (increase determinism)

            # sum components
            y_hat_diff = ar_component + sar_component + ma_component
            forecast.append(y_hat_diff)

            # update states
            if self.p > 0:
                ar_terms.append(y_hat_diff)
            if self.seasonal and self.P > 0 and self.m > 1:
                sar_terms.append(y_hat_diff)
            if q_total > 0:
                ma_terms.append(epsilon)

        # to pd.Series
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
        
        filename = Path(__file__).parent.parent.parent / f"models/arima/{self.name}.pkl"
        
        with open(filename, "wb") as f:
            dill.dump({'model': self.model, 'trace': self.trace}, f)
        
        return filename
    
    def load(self, filename: str = None):
        """
        Load the model from a file.
        
        Parameters:
        - filename: Path to the saved model. If None, uses the default naming convention.
        """
        if filename is None:
            filename = Path(__file__).parent.parent.parent / f"models/arima/{self.name}.pkl"

        with open(filename, 'rb') as f:
            data = dill.load(f)
            self.model = data['model']
            self.trace = data['trace']
