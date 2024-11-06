from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class Ensemble(ABC):
    @abstractmethod
    def ensemble(self, forecasts: List[pd.Series]) -> pd.Series:
        """
        Abstract method to combine forecasts.
        """
        pass
