from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List
import pandas as pd
from ..ensemble.ensemble import Ensemble

class RegressionEnsemble(Ensemble):
    """
    Ensemble method using a regression model to combine multiple forecasts.
    """

    def __init__(self):
        """
        Initializes the RegressionEnsemble with a Linear Regression model.
        """
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, forecasts: List[np.ndarray], actual: List[float], exog: List[np.ndarray] = None):
        """
        Train the regression model using the provided forecasts and actual values.
        
        Parameters:
            forecasts (List[np.ndarray]): List of forecasted values from different models.
            actual (List[float]): List of actual observed values.
            exog (List[np.ndarray]): List of exogenous features, or data augmentation vectors. Optional.
        """
        if not forecasts:
            raise ValueError("The 'forecasts' list is empty.")
        
        n_models = len(forecasts)
        n_samples = len(forecasts[0])

        # make sure all forecast arrays have the same number of samples
        for i, f in enumerate(forecasts):
            if len(f) != n_samples:
                raise ValueError(f"Forecast at index {i} has {len(f)} samples; expected {n_samples}.")

        # Stack horizontally to form the feature matrix
        X = np.column_stack(forecasts)  # Shape: (n_samples, n_models)

        # tack on exogenous variables to feature mat
        if exog is not None:
            if len(exog) != n_samples:
                raise ValueError(f"The number of exogenous samples ({len(exog)}) does not match the number of forecast samples, {n_samples}.")
            # stack exogenous variables horizontally
            exog_matrix = np.column_stack(exog)  # shape: (n_samples, n_exog_features)
            X = np.hstack((X, exog_matrix))     # shape: (n_samples, n_models + n_exog_features)

        y = np.array(actual)  # shape: (n_samples,)

        # fit the regression model
        self.model.fit(X, y)
        self.is_trained = True
        print("RegressionEnsemble: Training completed.")

    def ensemble(self, forecasts: np.ndarray, exog: np.ndarray = None) -> float:
        """
        Combine forecasts using the trained regression model to produce a single prediction.
        
        Parameters:
            forecasts (np.ndarray): Array of forecasted values from different models for a single sample.
            exog (np.ndarray): Array of exogenous features for the single sample. Optional.
        
        Returns:
            float: The ensemble prediction.
        """
        if not self.is_trained:
            raise RuntimeError("RegressionEnsemble: The model has not been trained yet.")

        if forecasts.ndim != 1:
            raise ValueError("RegressionEnsemble: 'forecasts' should be a 1D array.")

        X_new = forecasts.reshape(1, -1)  # shape: (1, n_models)

        if exog is not None:
            if exog.ndim != 1:
                raise ValueError("RegressionEnsemble: 'exog' should be a 1D array.")
            exog_new = exog.reshape(1, -1)      # shape: (1, n_exog_features)
            X_new = np.hstack((X_new, exog_new))  # combined Shape: (1, n_models + n_exog_features)

        prediction = self.model.predict(X_new)

        return prediction[0]
