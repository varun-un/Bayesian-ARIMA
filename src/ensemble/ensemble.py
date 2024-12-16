from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List

class Ensemble(ABC):
    """
    Ensembles the multiple forecasts into one from the hierarchical values.

    Ensembles just the final predictions, not the features or the entire forecasted series for simplicity.
    """

    @abstractmethod
    def ensemble(self, forecasts: np.ndarray, exog: np.ndarray = None) -> float:
        """
        Method to combine forecasts. Performs the forward pass of the ensemble.

        Pass in the order of daily, hourly, and minutely forecasts.

        Parameters:
        - forecasts (np.ndarray): Array of forecasted values from different models.
        - exog (np.ndarray): Array of exogenous features, or data augmentation vector. Optional.
        """
        pass

    @abstractmethod
    def train(self, forecasts: List[np.ndarray], actual: List[float], exog: List[np.ndarray] = None):
        """
        Method to train the ensemble. Will perform the backward pass of the ensemble, and use batch training.

        Pass in the order of daily, hourly, and minutely forecasts.

        For each parameter, corresponding indices in each list are the corresponding vectors and points for a single sample.
        The list represents the batch of samples.

        Parameters:
        - forecasts (List[np.ndarray]): List of forecasted values from different models.
        - actual (List[float]): List of actual values.
        - exog (List[np.ndarray]): List of exogenous features, or data augmentation vectors. Optional.
        """
        pass
