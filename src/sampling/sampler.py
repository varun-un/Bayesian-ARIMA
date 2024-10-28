from abc import ABC, abstractmethod
import pandas as pd

class Sampler(ABC):
    """
    Abstract class for sampling data.
    """


    @abstractmethod
    def sample(self, data: pd.Series) -> pd.Series:
        """
        Abstract method to sample data.
        """
        pass
