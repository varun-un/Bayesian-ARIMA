from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class Ensemble(ABC):
    """
    Ensembles the multiple forecasts into one from the hierarchical values.

    Ensembles just the final predictions, not the features or the entire forecasted series for simplicity.
    """

    @abstractmethod
    def ensemble(self, forecasts: List[float]) -> float:
        """
        Abstract method to combine forecasts. Performs the forward pass of the ensemble.
        """
        pass

    @abstractmethod
    def train(self, forecasts: List[float], actual: float):
        """
        Abstract method to train the ensemble.
        """
        pass
