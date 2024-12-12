# src/sampler.py

from .sampler import Sampler
import pandas as pd
import random

class MonteCarloSampler(Sampler):
    """
    Use uniform random sampling to sample windows of data.
    """

    def __init__(self, window_size: int, num_samples: int):
        """
        Initialize with window size, number of samples.

        Parameters:
        - window_size (int): The size of each sampled window.
        - num_samples (int): The number of windows to sample.
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        self.window_size = window_size
        self.num_samples = num_samples
        
        random.seed(42)

    def sample(self, data: pd.Series) -> pd.Series:
        """
        Randomly samples windows of data.

        Parameters:
        - data (pd.Series): The input time series data to sample from.

        Returns:
        - pd.Series: A concatenated series containing all sampled windows.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("Input data must be a pandas Series.")

        if self.window_size > len(data):
            raise ValueError("window_size cannot be greater than the length of the data.")

        sampled_windows = []
        for _ in range(self.num_samples):
            start = random.randint(0, len(data) - self.window_size)
            window = data.iloc[start:start + self.window_size]
            sampled_windows.append(window.reset_index(drop=True))
        
        # concatenate all sampled windows into a single Series
        sampled_data = pd.concat(sampled_windows, ignore_index=True)
        return sampled_data
