"""
Exploratory Data Analysis module for diabetes prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Using custom logger instead of standard logging
from ..core.config_loader import config_loader, DATASET_FILE, PLOTS_DIR


def run_eda(logger=None) -> pd.DataFrame:
    """Run comprehensive exploratory data analysis."""
    if logger is None:
        from ..utils.custom_logger import log_handler
        logger = log_handler
    
    logger.info("Starting Exploratory Data Analysis")
    
    # Load dataset
    df = pd.read_csv(DATASET_FILE)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    
    # Basic dataset information
    logger.info("Dataset Info:")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    # Class distribution
    class_dist = df['Diabetes_binary'].value_counts()
    class_pct = df['Diabetes_binary'].value_counts(normalize=True) * 100
    
    logger.info(f"Class distribution:\n{class_dist}")
    logger.info(f"Class percentage:\n{class_pct}")
    
    # Create visualizations
    create_eda_plots(df, logger)
    
    logger.info("Exploratory Data Analysis completed")
    return df


def create_eda_plots(df: pd.DataFrame, logger=None) -> None:
    """Create and save EDA visualizations."""
    if logger is None:
        from ..utils.custom_logger import log_handler
        logger = log_handler
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Class distribution plot
    plt.figure(figsize=(8, 6))
    df['Diabetes_binary'].value_counts().plot(kind='bar')
    plt.title('Distribution of Diabetes Binary Classes')
    plt.xlabel('Diabetes Binary (0=No, 1=Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation matrix
    plt.figure(figsize=(15, 12))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature distributions by diabetes status
    numeric_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(numeric_features):
        sns.boxplot(data=df, x='Diabetes_binary', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution by Diabetes Status')
        axes[i].set_xlabel('Diabetes Binary (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature importance visualization (correlation with target)
    target_corr = df.corr()['Diabetes_binary'].abs().sort_values(ascending=False)[1:]  # Exclude target itself
    
    plt.figure(figsize=(10, 8))
    target_corr.plot(kind='barh')
    plt.title('Feature Correlation with Diabetes Binary (Absolute Values)')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importance_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Age distribution by diabetes status
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Age', hue='Diabetes_binary')
    plt.title('Age Distribution by Diabetes Status')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Diabetes', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"EDA plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    from ..utils.custom_logger import log_handler
    
    run_eda(log_handler)