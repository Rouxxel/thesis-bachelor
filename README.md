# 🧠 Application of Machine Learning Techniques for the Early Detection of Diabetes  
### 🎓 Bachelor Thesis – Bachelor of Science (BSc.)  
**Department of Tech and Software**  
**Major: Software Engineering**  
**Author: Sebastian Russo**

This repository contains all the work and resources related to my Bachelor thesis on diabetes risk prediction using machine learning techniques. The project includes datasets, exploratory data analysis, model training and evaluation, visualizations, documentation, and presentation materials.

---

## 📘 Overview

This repository contains all materials related to my Bachelor thesis titled:  
**"Application of Machine Learning Techniques for the Early Detection of Diabetes: A Comparative Study of Classification Models"**

The study investigates and compares several machine learning models trained on health indicator datasets to predict diabetes risk. The process covers data cleaning, resampling, training, evaluation, and visualization of results for a comprehensive understanding of each model's performance.

The project has been restructured into a professional ML pipeline with:
- **Modular Architecture**: Clean separation of concerns with organized source code
- **JSON Configuration**: Centralized configuration management
- **Custom Logging**: Professional logging system with timestamped files
- **Build Scripts**: Cross-platform automation for easy setup and execution
- **Centralized Results**: All outputs organized in a single location

---

## 🚀 Quick Start

### Windows Users:
```cmd
# Setup and run everything
build.bat run

# Or step by step
build.bat           # Setup environment
python main.py      # Run pipeline
```

### Unix/Linux/macOS Users:
```bash
# Setup and run everything
chmod +x build.sh
./build.sh run

# Or step by step
./build.sh          # Setup environment
python main.py      # Run pipeline
```

### Using Make (any platform):
```bash
# Complete development workflow
make dev

# Or individual commands
make setup          # Setup directories
make install        # Install dependencies
make run-all        # Run complete pipeline
make clean          # Clean generated files
```

---

## 📁 Repository Structure

```
├── main.py                          # Main entry point - runs complete pipeline
├── setup.py                         # Project setup and directory creation
├── requirements.txt                 # Python dependencies
├── Makefile                         # Build automation (Unix-style)
├── build.sh                         # Build script (Unix/Linux/macOS)
├── build.bat                        # Build script (Windows)
├── README.md                        # This file
├── LICENSE                          # Project license
├── .gitignore                       # Git ignore rules
│
├── src/                             # Source code (NEW STRUCTURE)
│   ├── __init__.py
│   ├── core/                        # Core configuration and settings
│   │   ├── __init__.py
│   │   ├── config.json              # JSON configuration file
│   │   └── config_loader.py         # Configuration loader
│   ├── data/                        # Data processing and analysis
│   │   ├── __init__.py
│   │   └── eda.py                   # Exploratory Data Analysis
│   ├── models/                      # Machine Learning models
│   │   ├── __init__.py
│   │   ├── base_model.py            # Base model class
│   │   ├── decision_tree.py         # Decision Tree implementation
│   │   ├── logistic_regression.py   # Logistic Regression implementation
│   │   ├── random_forest.py         # Random Forest implementation
│   │   └── svm.py                   # Support Vector Machine implementation
│   ├── pipeline/                    # ML pipeline orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py              # Main pipeline logic
│   ├── scripts/                     # Utility scripts
│   │   ├── __init__.py
│   │   └── predict_diabetes.py      # Prediction script for new data
│   ├── utils/                       # Shared utilities
│   │   ├── __init__.py
│   │   ├── utils.py                 # Common utility functions
│   │   └── custom_logger.py         # Custom logging system
│   └── results/                     # Generated outputs (NEW)
│       ├── README.md                # Results documentation
│       ├── __init__.py
│       ├── logs/                    # Execution logs
│       ├── plots/                   # EDA and model visualizations
│       ├── trained_models/          # Saved ML models (.pkl)
│       └── analysis/                # Performance comparisons
│
├── dataset/                         # Dataset files
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv
│   ├── download_script_with_links.ipynb
│   └── other_considered_datasets/
│       ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│       ├── early_stage_diabetes_risk_prediction_dataset.csv
│       └── prima_indians_diabetes_database.csv
│
├── notebooks/                       # Jupyter notebooks (ORIGINAL)
│   └── dataset_analysis_EDA.ipynb   # Original EDA notebook
│
├── documentation/                   # Project documentation
│   ├── Bachelor_Thesis_Sebastian_Russo.docx
│   ├── Bachelor_Thesis_Proposal_Sebastian_Russo.docx
│   ├── Bachelor_Thesis_Proposal_long_version.docx
│   ├── Thesis_presentation_Sebastian_Russo.pptx
│   └── pdf_version/
│       ├── Bachelor_Thesis_Sebastian_Russo.pdf
│       └── Bachelor_Thesis_Proposal_Sebastian_Russo.pdf
│
└── diagrams_and_pictures/           # Visual artifacts (ORIGINAL)
    ├── EDA/                         # Correlation matrices, distributions
    ├── workflow/                    # Pipeline visualizations
    └── confusion_matrices/          # Model confusion matrices
```

---

## ⚙️ Project Architecture

### Modern ML Pipeline (NEW)

The project now features a professional ML pipeline architecture:

1. **Configuration Management**: JSON-based configuration in `src/core/config.json`
2. **Custom Logging**: Centralized logging system with timestamped files
3. **Modular Design**: Clean separation of data processing, models, and utilities
4. **Automated Pipeline**: Single command execution of the entire workflow
5. **Centralized Results**: All outputs organized in `src/results/`

### Pipeline Workflow

```
1. Data Loading & Preprocessing
   ├── Load dataset from configuration
   ├── Handle missing values and duplicates
   └── Split into train/test sets

2. Data Preparation
   ├── Apply SMOTE for class balancing
   ├── Feature scaling (when required)
   └── Data shuffling

3. Model Training (All Models)
   ├── Decision Tree
   ├── Logistic Regression
   ├── Random Forest
   └── Support Vector Machine

4. Model Evaluation
   ├── Performance metrics calculation
   ├── Cross-validation
   ├── Confusion matrix generation
   └── Model comparison

5. Results Generation
   ├── Save trained models (.pkl)
   ├── Generate visualizations (.png)
   ├── Create performance reports (.csv)
   └── Log detailed execution (.log)
```

### Original Notebook Workflow (PRESERVED)

Each original model notebook (`*_model.ipynb`) follows this pipeline:

1. **Logging Setup**: Logs results to console and corresponding `.log` file
2. **Dataset Loading**: Loads the primary BRFSS 2015 dataset
3. **Preprocessing**: Remove missing/duplicate records, train/test split
4. **Resampling**: Apply **SMOTE** to handle class imbalance
5. **Scaling**: Use **StandardScaler** for models requiring feature scaling
6. **Model Training**: Configure hyperparameters, train, and save as `.pkl`
7. **Evaluation**: Generate metrics, confusion matrices, and performance logs

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Automatic Setup (Recommended)

**Windows:**
```cmd
build.bat
```

**Unix/Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

**Using Make:**
```bash
make quick-setup
```

### Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup Directories:**
```bash
python setup.py
```

3. **Verify Configuration:**
```bash
python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.') / 'src')); from src.core.config_loader import config_loader; print('✅ Configuration loaded')"
```

### Requirements

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
imbalanced-learn==0.13.0
matplotlib==3.10.3
seaborn==0.13.2
joblib==1.4.2
```

---

## 🚀 Usage

### Complete Pipeline (NEW)

Run the entire ML pipeline with a single command:

```bash
# Complete pipeline
python main.py

# Or using build scripts
build.bat run        # Windows
./build.sh run       # Unix/Linux/macOS
make run-all         # Any platform with make
```

### Individual Components (NEW)

```bash
# Make predictions with trained models
python src/scripts/predict_diabetes.py --model decision_tree
python src/scripts/predict_diabetes.py --model random_forest --sample-data

# Available models: decision_tree, logistic_regression, random_forest, svm
```

### Original Notebooks (PRESERVED)

Run individual Jupyter notebooks for detailed analysis:

- `notebooks/dataset_analysis_EDA.ipynb`: Exploratory Data Analysis
- `decision_tree_model.ipynb`: Decision Tree model training
- `logistic_regression_model.ipynb`: Logistic Regression model training
- `random_forest_model.ipynb`: Random Forest model training
- `svm_model.ipynb`: SVM model training

> ⚠️ **Note:** Original notebooks are self-contained and must be run independently.

---

## 📊 Configuration

### JSON Configuration (NEW)

The project uses JSON-based configuration in `src/core/config.json`:

```json
{
    "dir_names": {
        "data_dir": "dataset",
        "trained_models": "src/results/trained_models",
        "logs_dir": "src/results/logs",
        "results_dir": "src/results/analysis",
        "plots_dir": "src/results/plots"
    },
    "model_params": {
        "decision_tree": {
            "class_weight": "balanced",
            "max_depth": 16,
            "random_state": 42
        },
        "logistic_regression": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "random_forest": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "max_depth": 16,
            "random_state": 42
        },
        "svm": {
            "class_weight": "balanced",
            "kernel": "rbf",
            "C": 0.01,
            "random_state": 42
        }
    },
    "training_params": {
        "train_test_split": 0.25,
        "cv_folds": 5,
        "random_state": 42
    },
    "logging": {
        "logging_level": "info",
        "log_file_name": "diabetes_prediction",
        "dir_name": "src/results/logs"
    }
}
```

---

## 📈 Results & Outputs

### Generated Files (NEW)

All outputs are centralized in `src/results/`:

**Logs** (`src/results/logs/`):
- `diabetes_prediction_YYYY-MM-DDTHH-MM-SS.log`: Timestamped execution logs

**Visualizations** (`src/results/plots/`):
- `class_distribution.png`: Dataset class distribution
- `correlation_matrix.png`: Feature correlation heatmap
- `feature_distributions.png`: Feature distributions by diabetes status
- `feature_importance_correlation.png`: Feature importance analysis
- `age_distribution.png`: Age distribution analysis
- `*_confusion_matrix.png`: Model-specific confusion matrices

**Trained Models** (`src/results/trained_models/`):
- `decision_tree_classifier.pkl`: Trained Decision Tree model
- `logistic_regression_classifier.pkl`: Trained Logistic Regression model
- `random_forest_classifier.pkl`: Trained Random Forest model
- `svm_classifier.pkl`: Trained SVM model

**Analysis** (`src/results/analysis/`):
- `model_comparison.csv`: Performance comparison of all models

### Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Classification Report**: Detailed per-class metrics
- **Confusion Matrix**: Visual performance representation
- **Cross-Validation**: 5-fold CV F1 scores
- **Training Time**: Model training duration

---

## 🧪 Models Implemented

### Machine Learning Models

1. **Logistic Regression**
   - Linear classification model
   - Requires feature scaling
   - Good baseline performance

2. **Decision Tree**
   - Tree-based classification
   - No scaling required
   - Interpretable results

3. **Random Forest**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance analysis

4. **Support Vector Machine (SVM)**
   - Kernel-based classification
   - Requires feature scaling
   - Effective for high-dimensional data

### Model Features (NEW)

- **Consistent Interface**: All models inherit from `BaseModel`
- **Automatic Scaling**: Applied when required by the model
- **Cross-Validation**: Built-in 5-fold cross-validation
- **Model Persistence**: Automatic saving as `.pkl` files
- **Performance Logging**: Detailed metrics and timing
- **Visualization**: Automatic confusion matrix generation

---

## 📦 Datasets

### Primary Dataset
- **`diabetes_binary_health_indicators_BRFSS2015.csv`**: Main dataset (253,680 records, 22 features)
  - Source: BRFSS 2015 survey data
  - Target: Binary diabetes classification (0=No, 1=Yes)
  - Features: Health indicators, demographics, lifestyle factors

### Alternative Datasets (Considered)
- **`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`**: Balanced version
- **`early_stage_diabetes_risk_prediction_dataset.csv`**: Early-stage risk prediction
- **`prima_indians_diabetes_database.csv`**: Benchmark dataset

---

## 🔧 Build System

### Cross-Platform Build Scripts (NEW)

**Windows (`build.bat`):**
```cmd
build.bat           # Setup environment
build.bat run       # Setup + run pipeline
build.bat clean     # Clean generated files
build.bat help      # Show usage
```

**Unix/Linux/macOS (`build.sh`):**
```bash
./build.sh          # Setup environment
./build.sh run      # Setup + run pipeline
./build.sh clean    # Clean generated files
./build.sh help     # Show usage
```

**Makefile (any platform with make):**
```bash
make help           # Show available commands
make setup          # Setup directories
make install        # Install dependencies
make run-all        # Run complete pipeline
make clean          # Clean generated files
make quick-setup    # Complete setup workflow
make dev            # Development workflow
```

### Build Features

- **Python Version Check**: Ensures Python 3.7+
- **Dependency Installation**: Automatic pip install
- **Configuration Validation**: Verifies JSON config
- **Dataset Verification**: Checks dataset availability
- **Error Handling**: Clear error messages and recovery
- **Cross-Platform**: Works on Windows, macOS, and Linux

---

## 🔍 Advanced Usage

### Custom Logger (NEW)

The project includes a professional logging system:

```python
from src.utils.custom_logger import log_handler

log_handler.info("Information message")
log_handler.warning("Warning message")
log_handler.error("Error message")
```

Features:
- **Timestamped Files**: Each run creates a unique log file
- **Dual Output**: Console and file logging
- **Configurable Levels**: Set via `config.json`
- **Professional Format**: Structured log messages

### Configuration Access (NEW)

Access configuration programmatically:

```python
from src.core.config_loader import config_loader

# Access model parameters
rf_params = config_loader["model_params"]["random_forest"]
n_estimators = rf_params["n_estimators"]

# Access training parameters
train_split = config_loader["training_params"]["train_test_split"]
```

### Making Predictions (NEW)

Use trained models for predictions:

```python
# Command line
python src/scripts/predict_diabetes.py --model random_forest

# Programmatic usage
from src.models.random_forest import RandomForestModel
import joblib

model = joblib.load("src/results/trained_models/random_forest_classifier.pkl")
predictions = model.predict(new_data)
```

---

## 🧹 Maintenance

### Cleaning Generated Files

**Using Build Scripts:**
```bash
build.bat clean     # Windows
./build.sh clean    # Unix/Linux/macOS
make clean          # Any platform
```

**Manual Cleanup:**
```bash
rm -rf src/results/logs/*.log
rm -rf src/results/plots/*.png
rm -rf src/results/analysis/*.csv
rm -rf src/results/trained_models/*.pkl
```

### Updating Configuration

Modify `src/core/config.json` to:
- Change model hyperparameters
- Adjust logging levels
- Update directory paths
- Modify training parameters

---

## 📈 Visuals & Diagrams

### Generated Visualizations (NEW)
- **EDA Plots**: Automatically generated in `src/results/plots/`
- **Confusion Matrices**: Model-specific performance visualization
- **Feature Analysis**: Correlation and importance plots

### Original Diagrams (PRESERVED)
- **`diagrams_and_pictures/EDA/`**: Original correlation matrices and distributions
- **`diagrams_and_pictures/workflow/`**: Pipeline and process visualizations
- **`diagrams_and_pictures/confusion_matrices/`**: Original model confusion matrices

---

## 📄 Documentation

### Project Documentation
- **`documentation/`**: Complete thesis and proposals
- **`src/results/README.md`**: Results directory documentation
- **Build Scripts**: Built-in help and usage examples

### Academic Materials
- **Bachelor_Thesis_Sebastian_Russo.pdf**: Complete thesis document
- **Thesis_presentation_Sebastian_Russo.pptx**: Defense presentation
- **Bachelor_Thesis_Proposal_Sebastian_Russo.pdf**: Original proposal

---

## 🤝 Contributing

### Development Workflow

1. **Setup Development Environment:**
```bash
make dev            # Complete development setup
```

2. **Make Changes:**
   - Modify source code in `src/`
   - Update configuration in `src/core/config.json`
   - Add new models by extending `BaseModel`

3. **Test Changes:**
```bash
python main.py      # Run complete pipeline
make clean          # Clean previous results
```

4. **Verify Results:**
   - Check logs in `src/results/logs/`
   - Review plots in `src/results/plots/`
   - Validate model outputs in `src/results/trained_models/`

---

## 📋 Troubleshooting

### Common Issues

**Python Not Found:**
- Install Python 3.7+ from [python.org](https://python.org)
- Ensure Python is in system PATH

**Dependencies Installation Failed:**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Dataset Not Found:**
- Ensure dataset is at: `dataset/diabetes_binary_health_indicators_BRFSS2015.csv`
- Check file permissions and path

**Configuration Errors:**
```bash
python -c "from src.core.config_loader import config_loader; print('Config OK')"
```

**Permission Denied (Unix/Linux/macOS):**
```bash
chmod +x build.sh
```

---

## 📊 Performance Benchmarks

### Expected Results

The pipeline typically achieves the following performance ranges:

- **Random Forest**: F1 Score ~0.85-0.90
- **Logistic Regression**: F1 Score ~0.80-0.85
- **Decision Tree**: F1 Score ~0.75-0.85
- **SVM**: F1 Score ~0.80-0.85

### Execution Time

- **Complete Pipeline**: 5-15 minutes (depending on hardware)
- **Individual Models**: 1-5 minutes each
- **EDA Generation**: 1-2 minutes

---

## 🎯 Future Enhancements

### Potential Improvements

1. **Additional Models**: Neural networks, gradient boosting
2. **Hyperparameter Tuning**: Grid search, random search
3. **Feature Engineering**: Advanced feature selection
4. **Model Interpretability**: SHAP values, LIME
5. **Web Interface**: Flask/Django web application
6. **API Endpoint**: REST API for predictions
7. **Docker Support**: Containerized deployment
8. **CI/CD Pipeline**: Automated testing and deployment

---

## 👤 Author

**Sebastian Russo**  
Bachelor Thesis – University of Europe for Applied Sciences  
Department of Tech and Software  
Major: Software Engineering

### Contact
- **Email**: [Your Email]
- **LinkedIn**: [Your LinkedIn]
- **GitHub**: [Your GitHub]

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **University of Europe for Applied Sciences** for academic support
- **BRFSS 2015** for providing the diabetes health indicators dataset
- **Scikit-learn** community for excellent machine learning tools
- **Open Source Community** for the libraries and tools used in this project

---

*This README provides comprehensive documentation for both the original thesis work and the modernized ML pipeline implementation.*