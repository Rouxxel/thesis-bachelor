"""
Random Forest model for diabetes prediction.
"""
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
# Using custom logger through base_model


class RandomForestModel(BaseModel):
    """Random Forest classifier for diabetes prediction."""
    
    def __init__(self, logger=None):
        super().__init__("Random Forest", logger)
    
    def create_model(self, **params):
        """Create Random Forest model."""
        return RandomForestClassifier(**params)
    
    def requires_scaling(self) -> bool:
        """Random Forest doesn't require feature scaling."""
        return False