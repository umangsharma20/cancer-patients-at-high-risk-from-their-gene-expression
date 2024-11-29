# Machine Learning Model for Gene Expression Classification

This repository contains a machine learning pipeline developed to classify gene expression data based on risk levels. Using a Random Forest classifier, along with essential data preprocessing, feature selection, and model evaluation techniques, this code is designed to generate predictions on test data and prepare submission-ready outputs.

## Dataset

- **Train Data:** Contains gene expression values and labels (1 for high risk, 0 for low risk).
- **Test Data:** Contains gene expression values without labels, used to generate predictions.
- **Sample Submission:** Provided to guide the format for output submissions.

## Code Overview

### 1. **Library Imports**
The script begins by importing essential libraries for data manipulation, scaling, feature selection, model training, and evaluation:
- `pandas`: For data loading and manipulation.
- `NumPy`: For array operations and numerical computations.
- `scikit-learn`: For machine learning utilities, including train-test splitting, standardization, feature selection, model training, and performance evaluation.

### 2. **Data Loading**
The code loads the training and test data from CSV files (`kaggle_train.csv` and `kaggle_test.csv`) into DataFrames.

### 3. **Data Preparation**
- **ID Extraction:** The ID column is extracted from the test data, as it does not contribute to the prediction.
- **Feature and Label Separation:** Features and labels are separated from the training data, assigning predictors to `X_train` and labels to `y_train`.

### 4. **Data Standardization**
Standardization is performed using `StandardScaler` to ensure all features are on the same scale, which is crucial for the performance of many machine learning algorithms.

### 5. **Feature Selection with RFE**
The **Recursive Feature Elimination (RFE)** process is used to select the most relevant features, reducing data dimensionality while retaining predictive information. A **RandomForestClassifier** is used as the estimator for RFE.

### 6. **Data Splitting**
Using `train_test_split`, the training data is split into a training and validation set, with 80% of data used for training and 20% for validation. This allows performance assessment on unseen data.

### 7. **Model Training**
The selected model, **RandomForestClassifier**, is trained using the training data. The model is set with 100 decision trees (`n_estimators=100`) and a fixed random seed for reproducibility.

### 8. **Model Evaluation on Validation Set**
The model predicts probabilities on the validation set, evaluating its performance using the **AUC-ROC** score, a common metric for binary classification tasks.

### 9. **Test Set Predictions**
The model generates probability-based predictions for the test data, indicating the likelihood of each instance belonging to the high-risk class (Labels = 1).

### 10. **Submission Preparation**
A structured DataFrame is created for submission, containing:
- **ID:** Unique identifier for each test instance.
- **Labels:** Probability of belonging to the high-risk class.

### 11. **Saving Submission**
The submission DataFrame is saved as `output.csv`, ready for submission or further analysis.

## Results

The **Validation AUC-ROC Score** obtained during evaluation provides insight into model performance. In this case, a score of **0.7645** indicates moderate classification performance.

## Usage

1. Clone this repository.
2. Ensure all dependencies are installed (`pandas`, `numpy`, `scikit-learn`).
3. Run the code to generate `output.csv` with predictions.

## License

This project is part of a learning assignment and is licensed for educational use.

