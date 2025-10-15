"""
Decision Tree model for diabetes prediction.
"""
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel
# Using custom logger through base_model


class DecisionTreeModel(BaseModel):
    """Decision Tree classifier for diabetes prediction."""
    
    def __init__(self, logger=None):
        super().__init__("Decision Tree", logger)
    
    def create_model(self, **params):
        """Create Decision Tree model."""
        return DecisionTreeClassifier(**params)
    
    def requires_scaling(self) -> bool:
        """Decision Tree doesn't require feature scaling."""
        return False