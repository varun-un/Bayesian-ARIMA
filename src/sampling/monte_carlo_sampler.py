from .sampler import Sampler
import pandas as pd
import random

class MonteCarloSampler(Sampler):
    """
    Use uniform random sampling to sample windows of data.
    """


    def __init__(self, window_size: int, num_samples: int):
        """
        Initialize with window size and number of samples.
        """
        self.window_size = window_size
        self.num_samples = num_samples

    def sample(self, data: pd.Series) -> pd.Series:
        """
        Randomly samples windows of data.
        """
        sampled_data = pd.Series(dtype=float)
        for _ in range(self.num_samples):
            start = random.randint(0, len(data) - self.window_size)
            window = data.iloc[start:start + self.window_size]
            sampled_data = sampled_data.append(window, ignore_index=True)
        return sampled_data
