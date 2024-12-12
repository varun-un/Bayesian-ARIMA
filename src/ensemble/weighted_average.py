from ..ensemble.ensemble import Ensemble
import pandas as pd
import numpy as np
from typing import List

class WeightedAverageEnsemble(Ensemble):
    def __init__(self, weights: List[float]):
        """
        Initialize with a list of weights.
        """
        self.weights = weights

    def train(self, forecasts: List[float], actual: float):
        return super().train(forecasts, actual)

    def ensemble(self, forecasts: List[float]) -> float:
        """
        Combine forecasts using weighted average.
        """
        if len(forecasts) != len(self.weights):
            raise ValueError("Number of forecasts and weights must match.")
        
        forecast = np.dot(forecasts, self.weights)

        return forecast
