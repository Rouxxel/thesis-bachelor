"""
Support Vector Machine model for diabetes prediction.
"""
from sklearn.svm import SVC
from .base_model import BaseModel
# Using custom logger through base_model


class SVMModel(BaseModel):
    """Support Vector Machine classifier for diabetes prediction."""
    
    def __init__(self, logger=None):
        super().__init__("SVM", logger)
    
    def create_model(self, **params):
        """Create SVM model."""
        return SVC(**params, probability=True)  # Enable probability estimates
    
    def requires_scaling(self) -> bool:
        """SVM requires feature scaling."""
        return True