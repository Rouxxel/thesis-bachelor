"""
Utility functions for the diabetes prediction ML pipeline.
"""
# Import custom logger
from .custom_logger import log_handler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import joblib
import time
from datetime import datetime


def get_custom_logger():
    """Get the custom logger instance."""
    from .custom_logger import log_handler
    return log_handler


def load_and_preprocess_data(data_path: Path, logger=None) -> tuple:
    """Load and preprocess the dataset."""
    if logger is None:
        logger = log_handler
        
    # Load dataset
    df = pd.read_csv(data_path)
    logger.info("Dataset successfully found and loaded")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.info(f"Found {missing_values} missing values. Applying imputation.")
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df = df_imputed
    else:
        logger.info("No missing values found; imputation not needed.")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows from the dataset.")
    
    # Separate features and target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    logger.info(f"Independent variables shape {X.shape}")
    logger.info(f"Dependent variables shape {y.shape}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int, logger=None) -> tuple:
    """Split data into training and testing sets."""
    if logger is None:
        logger = log_handler
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"X_train shape {X_train.shape} and X_test shape {X_test.shape}")
    logger.info(f"y_train shape {y_train.shape} and y_test shape {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, logger=None) -> tuple:
    """Apply SMOTE for handling class imbalance."""
    if logger is None:
        logger = log_handler
        
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Log class distribution after resampling
    class_dist = pd.Series(y_train_resampled).value_counts().to_dict()
    logger.info(f"Applied SMOTE. Class distribution after resampling: {class_dist}")
    
    return X_train_resampled, y_train_resampled


def apply_scaling(X_train: np.ndarray, X_test: np.ndarray, logger=None) -> tuple:
    """Apply feature scaling using StandardScaler."""
    if logger is None:
        logger = log_handler
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling applied using StandardScaler to training and test sets.")
    
    return X_train_scaled, X_test_scaled, scaler


def shuffle_data(X: np.ndarray, y: np.ndarray, logger=None) -> tuple:
    """Shuffle the training data."""
    if logger is None:
        logger = log_handler
        
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
    logger.info("Shuffled the training data after resampling")
    return X_shuffled, y_shuffled


def save_model(model, model_name: str, models_dir: Path, logger=None) -> None:
    """Save the trained model."""
    if logger is None:
        logger = log_handler
        
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Trained model '{model_name}.pkl' saved to directory {models_dir.name}")


def log_sample_predictions(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, logger=None, n_samples: int = 3) -> None:
    """Log sample predictions for verification."""
    if logger is None:
        logger = log_handler
        
    comparison_df = pd.DataFrame({
        'Actual': y_test[:n_samples],
        'Predicted': y_pred[:n_samples],
        'Probability_Class_1': y_proba[:n_samples, 1] if y_proba.ndim > 1 else y_proba[:n_samples]
    })
    logger.info(f"Comparison:\n{comparison_df}")


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, logger=None, operation: str = "Operation"):
        self.logger = logger if logger is not None else log_handler
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"{self.operation} completed in {elapsed_time:.4f} seconds.")