from ..ensemble.ensemble import Ensemble
import pandas as pd
from typing import List

class WeightedAverageEnsemble(Ensemble):
    def __init__(self, weights: List[float]):
        """
        Initialize with a list of weights.
        """
        self.weights = weights

    def ensemble(self, forecasts: List[pd.Series]) -> pd.Series:
        """
        Combine forecasts using weighted average.
        """
        if len(forecasts) != len(self.weights):
            raise ValueError("Number of forecasts and weights must match.")
        
        combined = pd.Series(0.0, index=forecasts[0].index)
        for forecast, weight in zip(forecasts, self.weights):
            combined += forecast * weight
        return combined
