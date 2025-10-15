#!/usr/bin/env python3
"""
Setup script for the Diabetes Prediction ML Pipeline.

This script creates necessary directories and validates the environment.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "src/results",
        "src/results/logs",
        "src/results/plots", 
        "src/results/trained_models",
        "src/results/analysis"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}/")


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version}")


def check_dataset():
    """Check if the main dataset exists."""
    dataset_path = Path("dataset/diabetes_binary_health_indicators_BRFSS2015.csv")
    if dataset_path.exists():
        print(f"✅ Dataset found: {dataset_path}")
    else:
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("   Please ensure the dataset is in the correct location")


def main():
    """Main setup function."""
    print("🧠 Diabetes Prediction ML Pipeline - Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Check dataset
    check_dataset()
    
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run complete pipeline: python main.py")
    print("3. Check results in: src/results/")
    print("   - Logs: src/results/logs/")
    print("   - Plots: src/results/plots/")
    print("   - Models: src/results/trained_models/")
    print("   - Analysis: src/results/analysis/")
    
    print("\n✅ Setup completed!")


if __name__ == "__main__":
    main()