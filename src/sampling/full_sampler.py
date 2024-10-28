from .sampler import Sampler
import pandas as pd

class FullSampler(Sampler):
    """
    A sampler that returns the entire dataset.
    """

    def sample(self, data: pd.Series) -> pd.Series:
        """
        Returns the entire dataset.
        """
        return data
