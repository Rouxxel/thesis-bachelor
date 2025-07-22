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

The aim of this thesis was to apply machine learning models to predict the risk of diabetes using various public datasets. The project involved:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Performance visualization (confusion matrices, accuracy, precision, recall, etc.)
- Documentation and presentation of findings

---

## 📁 Repository Structure

├── dataset/
│ ├── diabetes_binary_health_indicators_BRFSS2015.csv
│ ├── download_script_with_links.ipynb
│ └── other_considered_datasets/
│ ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│ ├── early_stage_diabetes_risk_prediction_dataset.csv
│ └── prima_indians_diabetes_database.csv
│
├── diagrams_and_pictures/
│ ├── EDA/
│ │ └── *.png # Correlation matrices, distributions, feature relations, etc.
│ ├── workflow/
│ │ ├── *.png # Visual representations of the pipeline/workflow
│ │ └── *.drawio
│ └── confusion_matrices/
│ └── *.png # Confusion matrices of the trained models
│
├── documentation/
│ ├── Bachelor_Thesis_Sebastian_Russo.docx
│ ├── Bachelor_Thesis_Proposal_Sebastian_Russo.docx
│ ├── Bachelor_Thesis_Proposal_long_version.docx
│ ├── Thesis_presentation_Sebastian_Russo.pptx
│ └── pdf_version/
│ ├── Bachelor_Thesis_Sebastian_Russo.pdf
│ └── Bachelor_Thesis_Proposal_Sebastian_Russo.pdf
│
├── .gitignore
├── requirements.txt
│
├── dataset_analysis_EDA.ipynb
│
├── decision_tree_model.ipynb
├── logistic_regression_model.ipynb
├── random_forest_model.ipynb
└── svm_model.ipynb

---

## ⚙️ Project Workflow

Each model script (`*_model.ipynb`) follows a consistent pipeline:

1. **Logging Setup**: Logs results to console and a corresponding `.log` file.
2. **Dataset Loading**: Loads the primary BRFSS 2015 dataset.
3. **Preprocessing**:
   - Remove missing/duplicate records.
   - Split into training and test sets: `X_train`, `Y_train`, `X_test`, `Y_test`.
4. **Resampling**: Apply **SMOTE** to handle class imbalance.
5. **Scaling**: Use **StandardScaler** for models that require feature scaling.
6. **Model Training**:
   - Configure model-specific hyperparameters.
   - Train on `X_train_resampled` or `X_train_scaled`.
   - Save trained model as `.pkl`.
7. **Evaluation**:
   - Predict with `X_test` or `X_test_scaled`.
   - Evaluate using multiple performance metrics (see below).
   - Generate and save confusion matrix plots.
   - Log all results for traceability.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Classification Report
- Confusion Matrix
- 5-Fold Cross-Validation F1 Scores
- Average Cross-Validation F1 Score
- Training Time

---

## 📈 Exploratory Data Analysis (EDA)

- Conducted using `dataset_analysis_EDA.ipynb`
- Visuals and findings are saved in `diagrams_and_pictures/EDA/`
- Includes feature correlation, class distribution, and health indicators impact.

---

## 🧪 Models Implemented

- `logistic_regression_model.ipynb`
- `decision_tree_model.ipynb`
- `random_forest_model.ipynb`
- `svm_model.ipynb`

Each model is trained on the **same dataset** with techniques tailored to each algorithm's characteristics.

---

## 🛠 Installation & Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```
### `requirements.txt` includes:

The following Python packages are required to run the notebooks:

```bash
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
imbalanced-learn==0.13.0
matplotlib==3.10.3
seaborn==0.13.2
joblib==1.4.2
```

---

## 📦 Datasets

The following datasets were used and considered:

- `diabetes_binary_health_indicators_BRFSS2015.csv`: Main dataset used for model training and testing.
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`: Balanced version of the main dataset.
- `early_stage_diabetes_risk_prediction_dataset.csv`: Alternative dataset for early-stage risk.
- `prima_indians_diabetes_database.csv`: A well-known benchmark dataset.

---

## ▶️ Running the Notebooks

> ⚠️ **Important:** Each notebook is self-contained and must be run independently. They do not share variables or intermediate results across scripts.

### 🔍 Exploratory Data Analysis

- `dataset_analysis_EDA.ipynb`:  
  Performs initial data inspection, correlation analysis, feature distribution visualization, and outlier detection. Useful for understanding the dataset before model training.

### 🤖 Model Training & Evaluation

Each of the following notebooks trains a different machine learning model on the same dataset. The goal is to compare performance across techniques using consistent preprocessing and evaluation methods:

- `logistic_regression_model.ipynb`
- `decision_tree_model.ipynb`
- `random_forest_model.ipynb`
- `svm_model.ipynb`

Each notebook follows the same structured workflow:

1. **Logging Setup**:  
   Configures a log handler to capture all steps and results in both the console and a `.log` file.

2. **Dataset Loading**:  
   Loads `diabetes_binary_health_indicators_BRFSS2015.csv` from the `dataset/` directory.

3. **Preprocessing**:
   - Removes missing or duplicate entries
   - Splits the data into training and test sets: `X_train`, `Y_train`, `X_test`, `Y_test`

4. **Resampling**:  
   Applies **SMOTE** to balance the training data.

5. **Scaling** (if applicable):  
   Uses **StandardScaler** for models that benefit from feature normalization (e.g., logistic regression, SVM).

6. **Model Training**:  
   Trains the model using the processed data and saves it as a `.pkl` file for future inference.

7. **Evaluation**:  
   Predicts on the test set and logs key metrics including:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC
   - Classification Report
   - Confusion Matrix (saved as PNG)
   - 5-Fold Cross-Validation F1 Scores
   - Average Cross-Validation F1 Score
   - Training Time

Each model’s results are stored in:
- `.log` files (console + file logging)
- `.pkl` files (trained model)
- `.png` files (confusion matrices and other graphs)

---

## 📈 Visuals & Diagrams

You can find all the visual artifacts used in the thesis under `diagrams_and_pictures/`:

- `EDA/`: Heatmaps, correlation matrices, and feature importance plots.
- `workflow/`: Visual explanations of the ML pipeline and process.
- `confusion_matrices/`: Confusion matrices of the trained classifiers.

---

## 📄 Documentation

Documentation can be found in the documentation/ folder, including:
- Full thesis and proposals (.docx and .pdf)
- Presentation slides (.pptx)
- Research methodology and findings

---

## 👤 Author

**Sebastian Russo**  
Bachelor Thesis – [University of Europe for Applied Sciences]
