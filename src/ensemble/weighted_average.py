import numpy as np
from typing import List
from src.ensemble import Ensemble

class WeightedAverageEnsemble(Ensemble):
    """
    Ensemble method using a weighted average of multiple forecasts.
    """

    def __init__(self, weights: List[float]):
        """
        Initializes the WeightedAverageEnsemble with specified weights.
        
        Parameters:
            weights (List[float]): List of weights for each forecasted model.
                                   The length of weights should match the number of forecast models.
        """
        self.weights = np.array(weights)
        self.n_models = len(weights)
        self.is_trained = False

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("WeightedAverageEnsemble: All weights must be finite numbers.")

        if not np.any(self.weights):
            raise ValueError("WeightedAverageEnsemble: At least one weight must be non-zero.")

    def train(self, forecasts: List[np.ndarray], actual: List[float], exog: List[np.ndarray] = None):
        
        pass

    def ensemble(self, forecasts: np.ndarray, exog: np.ndarray = None) -> float:
        """
        Combine forecasts by computing the weighted average.
        
        Parameters:
            forecasts (np.ndarray): 1D array of forecasted values from different models for a single sample.
            exog (np.ndarray): Ignored for WeightedAverageEnsemble.
        
        Returns:
            float: The weighted average prediction.
        """

        if forecasts.ndim != 1:
            raise ValueError("WeightedAverageEnsemble: 'forecasts' should be a 1D array.")

        if len(forecasts) != self.n_models:
            raise ValueError(f"WeightedAverageEnsemble: Number of forecasts ({len(forecasts)}) does not match number of weights ({self.n_models}).")

        weighted_avg = np.dot(forecasts, self.weights)

        return weighted_avg
    
    def set_weights(self, weights: List[float]):
        """
        Set the ensemble weights.
        
        Parameters:
            weights (List[float]): List of weights for each forecasted model.
                                   The length of weights should match the number of forecast models.
        """
        self.weights = np.array(weights)

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("WeightedAverageEnsemble: All weights must be finite numbers.")

        if not np.any(self.weights):
            raise ValueError("WeightedAverageEnsemble: At least one weight must be non-zero.")
