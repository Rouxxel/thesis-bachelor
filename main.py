#!/usr/bin/env python3
"""
Main entry point for the Diabetes Prediction ML Pipeline.

This script runs the complete machine learning pipeline including:
1. Exploratory Data Analysis (EDA)
2. Data preprocessing and preparation
3. Training of multiple ML models (Decision Tree, Logistic Regression, Random Forest, SVM)
4. Model evaluation and comparison
5. Results visualization and saving

Author: Sebastian Russo
Project: Application of Machine Learning Techniques for the Early Detection of Diabetes
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.pipeline.pipeline import DiabetesPredictionPipeline


def main():
    """Main function to run the complete diabetes prediction pipeline."""
    print("🧠 Diabetes Prediction ML Pipeline")
    print("=" * 50)
    print("Author: Sebastian Russo")
    print("Project: Application of Machine Learning Techniques for the Early Detection of Diabetes")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = DiabetesPredictionPipeline()
        pipeline.run_complete_pipeline()
        
        print("\n✅ Pipeline completed successfully!")
        print("📊 Check the following directories for results:")
        print("   - src/results/logs/: Training logs and detailed results")
        print("   - src/results/plots/: EDA visualizations and confusion matrices")
        print("   - src/results/analysis/: Model comparison CSV")
        print("   - src/results/trained_models/: Saved model files (.pkl)")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print("Check the src/results/logs/ directory for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()