# Results Directory

This directory contains all generated outputs from the Diabetes Prediction ML Pipeline.

## 📁 Directory Structure

```
src/results/
├── README.md                    # This file
├── __init__.py                  # Python package marker
├── logs/                        # Log files from pipeline runs
│   └── diabetes_prediction_YYYY-MM-DDTHH-MM-SS.log
├── plots/                       # EDA visualizations and model plots
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   ├── feature_importance_correlation.png
│   ├── age_distribution.png
│   └── *_confusion_matrix.png   # Model-specific confusion matrices
├── trained_models/              # Saved ML models (.pkl files)
│   ├── decision_tree_classifier.pkl
│   ├── logistic_regression_classifier.pkl
│   ├── random_forest_classifier.pkl
│   └── svm_classifier.pkl
└── analysis/                    # Analysis results and comparisons
    └── model_comparison.csv     # Performance comparison of all models
```

## 📊 File Descriptions

### Logs (`logs/`)
- **Purpose**: Detailed execution logs from pipeline runs
- **Format**: Timestamped log files with INFO, WARNING, ERROR messages
- **Usage**: Debugging, monitoring pipeline progress, performance tracking

### Plots (`plots/`)
- **Purpose**: Data visualizations and model performance plots
- **EDA Plots**: Data distribution, correlations, feature analysis
- **Model Plots**: Confusion matrices for each trained model
- **Format**: High-resolution PNG images (300 DPI)

### Trained Models (`trained_models/`)
- **Purpose**: Serialized ML models ready for prediction
- **Format**: Python pickle files (.pkl)
- **Usage**: Load models for making predictions on new data
- **Models**: Decision Tree, Logistic Regression, Random Forest, SVM

### Analysis (`analysis/`)
- **Purpose**: Comparative analysis and summary results
- **model_comparison.csv**: Performance metrics for all models
- **Usage**: Model selection, performance reporting

## 🚀 Usage

All files in this directory are automatically generated when you run:

```bash
python main.py
```

## 🧹 Cleanup

To clean all generated results:

```bash
# Remove all generated files (keep directory structure)
rm -rf src/results/logs/* src/results/plots/* src/results/trained_models/* src/results/analysis/*
```

## 📝 Notes

- Log files are timestamped to avoid conflicts between runs
- All paths are configured in `src/core/config.json`
- Results are organized for easy access and analysis
- Directory structure is created automatically if missing