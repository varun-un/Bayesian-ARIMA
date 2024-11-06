from .base_ensemble import BaseEnsemble
import pandas as pd
from typing import List
from sklearn.linear_model import LinearRegression
import numpy as np

class RegressionEnsemble(BaseEnsemble):
    def __init__(self):
        """
        Initialize the regression model.
        """
        self.model = LinearRegression()

    def ensemble(self, forecasts: List[pd.Series], actual: pd.Series) -> pd.Series:
        """
        Train a regression model to combine forecasts.
        Assumes 'actual' is provided for training.
        """
        # Create feature matrix
        X = np.column_stack([fc.values for fc in forecasts])
        y = actual.values
        
        # Fit regression model
        self.model.fit(X, y)
        
        # Predict using the trained model
        ensemble_values = self.model.predict(X)
        return pd.Series(ensemble_values, index=actual.index)
