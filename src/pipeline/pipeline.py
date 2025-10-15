"""
Main ML pipeline for diabetes prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
# Using custom logger instead of standard logging
from typing import Dict, Any

from ..core.config_loader import (
    config_loader, DATASET_FILE, LOGS_DIR, MODELS_DIR, RESULTS_DIR, 
    TRAIN_TEST_SPLIT, RANDOM_STATE
)
from ..utils.utils import (
    get_custom_logger, load_and_preprocess_data, split_data, 
    apply_smote, apply_scaling, shuffle_data
)
from ..utils.custom_logger import log_handler
from ..data.eda import run_eda
from ..models.decision_tree import DecisionTreeModel
from ..models.logistic_regression import LogisticRegressionModel
from ..models.random_forest import RandomForestModel
from ..models.svm import SVMModel


class DiabetesPredictionPipeline:
    """Main pipeline for diabetes prediction ML models."""
    
    def __init__(self):
        self.logger = log_handler
        self.models = {
            'decision_tree': DecisionTreeModel,
            'logistic_regression': LogisticRegressionModel,
            'random_forest': RandomForestModel,
            'svm': SVMModel
        }
        self.results = {}
        
    def run_eda(self) -> pd.DataFrame:
        """Run exploratory data analysis."""
        self.logger.info("=" * 50)
        self.logger.info("STARTING EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 50)
        
        df = run_eda(self.logger)
        return df
    
    def prepare_data(self):
        """Load and prepare data for training."""
        self.logger.info("=" * 50)
        self.logger.info("PREPARING DATA")
        self.logger.info("=" * 50)
        
        # Load and preprocess data
        X, y = load_and_preprocess_data(DATASET_FILE, self.logger)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, TRAIN_TEST_SPLIT, RANDOM_STATE, self.logger
        )
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, self.logger)
        
        # Shuffle data
        X_train_resampled, y_train_resampled = shuffle_data(
            X_train_resampled, y_train_resampled, self.logger
        )
        
        return X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train and evaluate a single model."""
        self.logger.info("=" * 50)
        self.logger.info(f"TRAINING {model_name.upper()} MODEL")
        self.logger.info("=" * 50)
        
        # Create model instance
        model_class = self.models[model_name]
        model = model_class(self.logger)
        
        # Prepare training data
        X_train_final = X_train.copy()
        X_test_final = X_test.copy()
        
        # Apply scaling if required
        if model.requires_scaling():
            X_train_final, X_test_final, scaler = apply_scaling(
                X_train_final, X_test_final, self.logger
            )
            # Shuffle again after scaling
            X_train_final, y_train = shuffle_data(X_train_final, y_train, self.logger)
        
        # Get model parameters
        params = config_loader["model_params"][model_name]
        
        # Train model
        model.train(X_train_final, y_train, **params)
        
        # Evaluate model
        results = model.evaluate(X_test_final, y_test, X_train_final, y_train)
        
        # Save model and plots
        model.save_model()
        model.save_confusion_matrix()
        
        return results
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train and evaluate all models."""
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled = self.prepare_data()
        
        # Train each model
        all_results = {}
        for model_name in self.models.keys():
            try:
                results = self.train_model(
                    model_name, X_train_resampled, y_train_resampled, X_test, y_test
                )
                all_results[model_name] = results
                self.logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return all_results
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Compare results across all models."""
        self.logger.info("=" * 50)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("=" * 50)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': model_results['accuracy'],
                'Precision': model_results['precision'],
                'Recall': model_results['recall'],
                'F1 Score': model_results['f1_score'],
                'ROC AUC': model_results['roc_auc'],
                'CV F1 Mean': model_results['cv_mean'] if model_results['cv_mean'] else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        self.logger.info("Model Performance Comparison:")
        self.logger.info(f"\n{comparison_df.to_string(index=False, float_format='%.4f')}")
        
        # Save comparison to CSV
        comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        self.logger.info(f"Model comparison saved to {RESULTS_DIR / 'model_comparison.csv'}")
        
        # Log best model
        best_model = comparison_df.iloc[0]['Model']
        best_f1 = comparison_df.iloc[0]['F1 Score']
        self.logger.info(f"Best performing model: {best_model} (F1 Score: {best_f1:.4f})")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete ML pipeline."""
        self.logger.info("STARTING COMPLETE DIABETES PREDICTION PIPELINE")
        self.logger.info("=" * 70)
        
        try:
            # Step 1: EDA
            self.run_eda()
            
            # Step 2: Train all models
            results = self.train_all_models()
            
            # Step 3: Compare models
            if results:
                self.compare_models(results)
            
            self.logger.info("=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main function to run the pipeline."""
    pipeline = DiabetesPredictionPipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()