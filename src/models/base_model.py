"""
Base model class for diabetes prediction models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from pathlib import Path
# Using custom logger instead of standard logging
from abc import ABC, abstractmethod
from ..utils.utils import Timer, save_model, log_sample_predictions
from ..core.config_loader import config_loader, MODELS_DIR, PLOTS_DIR


class BaseModel(ABC):
    """Base class for all ML models."""
    
    def __init__(self, model_name: str, logger=None):
        from ..utils.custom_logger import log_handler
        self.model_name = model_name
        self.logger = logger if logger is not None else log_handler
        self.model = None
        self.is_trained = False
        self.results = {}
    
    @abstractmethod
    def create_model(self, **params):
        """Create the model instance."""
        pass
    
    @abstractmethod
    def requires_scaling(self) -> bool:
        """Return True if the model requires feature scaling."""
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **params) -> None:
        """Train the model."""
        self.model = self.create_model(**params)
        
        with Timer(self.logger, f"Model training"):
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.logger.info(f"Trained {self.model_name} model with parameters: {params}")
    
    def predict(self, X_test: np.ndarray) -> tuple:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        y_pred = self.model.predict(X_test)
        
        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)
        elif hasattr(self.model, 'decision_function'):
            # For SVM, convert decision function to probabilities
            decision_scores = self.model.decision_function(X_test)
            y_proba = np.column_stack([1 - decision_scores, decision_scores])
        else:
            y_proba = np.column_stack([1 - y_pred, y_pred])
        
        return y_pred, y_proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, X_train: np.ndarray = None, y_train: np.ndarray = None) -> dict:
        """Evaluate the model performance."""
        y_pred, y_proba = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC
        if y_proba.ndim > 1:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba)
        
        # Cross-validation (if training data provided)
        cv_scores = None
        cv_mean = None
        if X_train is not None and y_train is not None:
            cv = StratifiedKFold(n_splits=config_loader["training_params"]["cv_folds"], shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='f1_weighted')
            cv_mean = cv_scores.mean()
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Log results
        self.log_results()
        
        # Log sample predictions
        log_sample_predictions(y_test, y_pred, y_proba, self.logger)
        
        return self.results
    
    def log_results(self) -> None:
        """Log evaluation results."""
        self.logger.info("=== MODEL EVALUATION RESULTS ===")
        self.logger.info(f"Accuracy: {self.results['accuracy']:.4f}")
        self.logger.info(f"Precision: {self.results['precision']:.4f}")
        self.logger.info(f"Recall: {self.results['recall']:.4f}")
        self.logger.info(f"F1 Score: {self.results['f1_score']:.4f}")
        self.logger.info(f"ROC AUC: {self.results['roc_auc']:.4f}")
        
        if self.results['cv_scores'] is not None:
            self.logger.info(f"Cross-validation F1 scores: {self.results['cv_scores']}")
            self.logger.info(f"Average CV F1 Score: {self.results['cv_mean']:.4f}")
        
        self.logger.info("Confusion Matrix:")
        self.logger.info(f"\n{self.results['confusion_matrix']}")
        
        self.logger.info("Classification Report:")
        self.logger.info(f"\n{self.results['classification_report']}")
    
    def save_confusion_matrix(self) -> None:
        """Save confusion matrix plot."""
        if 'confusion_matrix' not in self.results:
            self.logger.warning("No confusion matrix available to save")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plot_path = PLOTS_DIR / f'{self.model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {plot_path}")
    
    def save_model(self) -> None:
        """Save the trained model."""
        if not self.is_trained:
            self.logger.warning("Model is not trained, cannot save")
            return
        
        model_filename = f"{self.model_name.lower().replace(' ', '_')}_classifier"
        save_model(self.model, model_filename, MODELS_DIR, self.logger)