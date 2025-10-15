"""
Logistic Regression model for diabetes prediction.
"""
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel
# Using custom logger through base_model


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier for diabetes prediction."""
    
    def __init__(self, logger=None):
        super().__init__("Logistic Regression", logger)
    
    def create_model(self, **params):
        """Create Logistic Regression model."""
        return LogisticRegression(**params)
    
    def requires_scaling(self) -> bool:
        """Logistic Regression requires feature scaling."""
        return True