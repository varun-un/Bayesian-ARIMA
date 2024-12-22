from sklearn.linear_model import SGDRegressor
import numpy as np
from typing import List
from src.ensemble import Ensemble  # Assuming this is a valid import

class RegressionEnsemble(Ensemble):
    """
    Ensemble method using a regression model to combine multiple forecasts.
    Supports online learning by allowing incremental training with new data batches.
    """

    def __init__(self, **sgd_params):
        """
        Initializes the RegressionEnsemble with an SGD Regressor for online learning.

        Parameters:
            **sgd_params: Arbitrary keyword arguments for SGDRegressor.
                          For example, you can set 'learning_rate', 'eta0', etc.
        """
        self.model = SGDRegressor(**sgd_params)
        self.is_trained = False

    def train(self, forecasts: List[np.ndarray], actual: List[float], exog: List[np.ndarray] = None):
        """
        Train the regression model using the provided forecasts and actual values.
        Supports incremental training for online learning.

        Parameters:
            forecasts (List[np.ndarray]): List of forecasted values from different models.
                                         Each array should be of shape (n_samples,).
            actual (List[float]): List of actual observed values. Should be of length n_samples.
            exog (List[np.ndarray]): List of exogenous features for each sample.
                                      Each array should be of shape (n_exog_features,).
                                      Optional.
        """
        if not forecasts:
            raise ValueError("The 'forecasts' list is empty.")
        
        n_models = len(forecasts)
        n_samples = len(forecasts[0])

        # Ensure all forecast arrays have the same number of samples
        for i, f in enumerate(forecasts):
            if len(f) != n_samples:
                raise ValueError(f"Forecast at index {i} has {len(f)} samples; expected {n_samples}.")

        # Stack forecasts horizontally to form the feature matrix
        X = np.column_stack(forecasts)  # Shape: (n_samples, n_models)

        # Append exogenous variables if provided
        if exog is not None:
            if len(exog) != n_samples:
                raise ValueError(f"The number of exogenous samples ({len(exog)}) does not match the number of forecast samples ({n_samples}).")
            
            # Each exog[i] should be a 1D array of shape (n_exog_features,)
            exog_matrix = np.vstack(exog)  # Shape: (n_samples, n_exog_features)
            X = np.hstack((X, exog_matrix))  # Shape: (n_samples, n_models + n_exog_features)

        y = np.array(actual)  # Shape: (n_samples,)

        # Incrementally fit the model
        self.model.partial_fit(X, y)
        self.is_trained = True
        print("RegressionEnsemble: Incremental training completed.")

    def ensemble(self, forecasts: np.ndarray, exog: np.ndarray = None) -> float:
        """
        Combine forecasts using the trained regression model to produce a single prediction.

        Parameters:
            forecasts (np.ndarray): Array of forecasted values from different models for a single sample.
                                     Should be of shape (n_models,).
            exog (np.ndarray): Array of exogenous features for the single sample.
                               Should be of shape (n_exog_features,). Optional.

        Returns:
            float: The ensemble prediction.
        """
        if not self.is_trained:
            raise RuntimeError("RegressionEnsemble: The model has not been trained yet.")

        if forecasts.ndim != 1:
            raise ValueError("RegressionEnsemble: 'forecasts' should be a 1D array.")

        X_new = forecasts.reshape(1, -1)  # Shape: (1, n_models)

        if exog is not None:
            if exog.ndim != 1:
                raise ValueError("RegressionEnsemble: 'exog' should be a 1D array.")
            exog_new = exog.reshape(1, -1)  # Shape: (1, n_exog_features)
            X_new = np.hstack((X_new, exog_new))  # Shape: (1, n_models + n_exog_features)

        prediction = self.model.predict(X_new)

        return prediction[0]
