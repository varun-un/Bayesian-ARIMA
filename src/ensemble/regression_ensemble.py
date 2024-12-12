import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod
from ..ensemble.ensemble import Ensemble


class RegressionEnsemble(Ensemble):
    """
    Online Ridge Regression Ensemble Method

    Based off: https://www.ibm.com/topics/ridge-regression

    Use ridge regression instead of lasso regression to avoid sparsity in the weights - want to consider all models.
    """

    def __init__(self, regularization: float = 1e-4):
        """
        Initialize the Online RegressionEnsemble.
        
        Parameters:
        - regularization (float): L2 regularization parameter to prevent overfitting (default=1e-4) 
        """
        self.weights = None
        self.regularization = regularization
        
        # cumulative statistics for incremental learning
        # storing these allows easy updates to weights using online algorithms
        self.total_observations = 0
        self.cumulative_XtX = None  # Σ (X^T * X)
        self.cumulative_Xty = None  # Σ (X^T * y)
    
    def train(self, forecasts: List[float], actual: float):
        """
        Incrementally train the regression using online learning.
        
        Parameters:
        - forecasts (List[float]): A list of prediction values from different models
        - actual (float): The ground truth value for the current time step
        """
        # convert forecasts to a numpy array (feature vector)
        X = np.array(forecasts).reshape(-1, 1) # shape: (num_models, 1)
        y = actual  

        # cumulative statistics
        if self.cumulative_XtX is None:
            self.cumulative_XtX = X @ X.T  # shape: (num_models, num_models)
            self.cumulative_Xty = X.flatten() * y  # shape: (num_models,)
        else:
            self.cumulative_XtX += X @ X.T
            self.cumulative_Xty += X.flatten() * y

        self.total_observations += 1

        # add L2 regularization to the diagonal of the XtX matrix - ridge regression
        regularization_matrix = np.eye(X.shape[0]) * self.regularization

        previous_weights = self.weights

        try:
            # weights = (XtX + lambda * I)^-1 * Xty     (from the paper/website)
            XtX_reg = self.cumulative_XtX + regularization_matrix
            self.weights = np.linalg.solve(XtX_reg, self.cumulative_Xty)
            
            # ensure weights are non-negative and normalize to sum to 1
            self.weights = np.maximum(self.weights, 0)
            weight_sum = np.sum(self.weights)
            if weight_sum > 0:
                self.weights /= weight_sum
            else:
                # if weights are 0, default to uniform weights
                self.weights = np.ones(X.shape[0]) / X.shape[0]

        except np.linalg.LinAlgError:
            print("Error occurred with ridge regression, using previous weights.")
            self.weights = previous_weights

    def ensemble(self, forecasts: List[float]) -> float:
        """
        Combine forecasts using the learned optimal weights.
        
        Parameters:
        - forecasts (List[float]): A list of prediction values from different models
        
        Returns:
        - float: The combined forecast using learned weights
        """

        if self.weights is None:
            raise ValueError("Ensemble must be trained before making predictions.")
        
        # check that number of forecasts matches training
        if len(forecasts) != len(self.weights):
            raise ValueError("Number of forecasts must match number of models used during training.")
        
        # forecasts to a numpy array
        X = np.array(forecasts)  # Shape: (num_models,)
        
        # weighted sum
        ensemble_forecast = np.dot(self.weights, X)
        
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
