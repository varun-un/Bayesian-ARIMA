import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
from ..ensemble.ensemble import Ensemble


class RegressionEnsemble(Ensemble):
    """
    Online Ridge Regression Ensemble Method

    Based off: https://www.ibm.com/topics/ridge-regression
    """

    def __init__(self, 
                 learning_rate: float = 0.1, 
                 regularization: float = 1e-4):
        """
        Initialize the Online RegressionEnsemble.
        
        Parameters:
        - learning_rate (float): Controls the step size of weight updates (default=0.1) 
        - regularization (float): L2 regularization parameter to prevent overfitting (default=1e-4) 
        """
        self.weights = None
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # cumulative statistics for incremental learning
        # storing these are what the online algorithm affects, so we can update the weights easily
        self.total_observations = 0
        self.cumulative_XtX = None  # Σ (X^T * X)
        self.cumulative_Xty = None  # Σ (X^T * y)
    
    def train(self, forecasts: List[pd.Series], actual: pd.Series):
        """
        Incrementally train the regression using online learning.
        
        Parameters:
        - forecasts (List[pd.Series]): A list of prediction series from different models
        - actual (pd.Series): The ground truth time series
        """
        # align all the data series so they can be concurrently indexed (corresponding points)
        aligned_forecasts = []
        for forecast in forecasts:
            aligned_forecast = forecast.reindex(actual.index).fillna(0)
            aligned_forecasts.append(aligned_forecast)
        
        X = np.column_stack([forecast.values for forecast in aligned_forecasts])
        y = actual.values
        
        # if first training iteration, initialize weights and cumulative statistics
        if self.weights is None:
            self.weights = np.ones(X.shape[1]) / X.shape[1]
            self.cumulative_XtX = X.T @ X
            self.cumulative_Xty = X.T @ y
        else:
            # Update cumulative statistics
            self.cumulative_XtX += X.T @ X
            self.cumulative_Xty += X.T @ y
        
        self.total_observations += X.shape[0]
        
        # adding small L2 regularization to the diagonal of the XtX matrix - ridge regression
        regularization_matrix = np.eye(X.shape[1]) * self.regularization
        
        try:
            # Compute weights using regularized inverse
            self.weights = np.linalg.solve(
                self.cumulative_XtX + regularization_matrix, 
                self.cumulative_Xty
            )
            
            # keep weights are non-negative and sum to 1
            self.weights = np.maximum(self.weights, 0)
            self.weights /= np.sum(self.weights)
        except np.linalg.LinAlgError:
            print("Error detected. Maintaining previous weights.")
    
    def ensemble(self, forecasts: List[pd.Series]) -> pd.Series:
        """
        Combine forecasts using the learned optimal weights.
        
        Parameters:
        - forecasts (List[pd.Series]): A list of prediction series from different models
        
        Returns:
        - pd.Series: The combined forecast using learned weights
        """
        # Check if weights have been learned
        if self.weights is None:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # Check that number of forecasts matches training
        if len(forecasts) != len(self.weights):
            raise ValueError("Number of forecasts must match number of models used during training.")
        
        # Align forecasts and apply weights
        weighted_forecasts = []
        for forecast, weight in zip(forecasts, self.weights):
            weighted_forecasts.append(forecast * weight)
        
        # Sum the weighted forecasts
        ensemble_forecast = pd.Series(
            np.sum(np.column_stack([f.values for f in weighted_forecasts]), axis=1),
            index=forecasts[0].index
        )
        
        return ensemble_forecast
    
    def get_weights(self) -> np.ndarray:
        """
        Retrieve the learned weights for each model.
        
        Returns:
        - np.ndarray: The learned weights for each model
        """
        if self.weights is None:
            raise ValueError("Ensemble must be trained before accessing weights.")
        return self.weights
    
    def reset(self):
        """
        Reset the ensemble to its initial state.
        Useful for starting over or changing datasets completely.
        """
        self.weights = None
        self.total_observations = 0
        self.cumulative_XtX = None
        self.cumulative_Xty = None