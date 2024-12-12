from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class Ensemble(ABC):
    @abstractmethod
    def ensemble(self, forecasts: List[pd.Series]) -> pd.Series:
        """
        Abstract method to combine forecasts. Performs the forward pass of the ensemble.
        """
        pass

    @abstractmethod
    def train(self, forecasts: List[pd.Series], actual: pd.Series):
        """
        Abstract method to train the ensemble.
        """
        pass
