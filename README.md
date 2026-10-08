# Telco Customer Churn Prediction Pipeline

## 📌 Project Overview
An end-to-end Machine Learning pipeline to predict customer churn using a telecom dataset (50,000+ records). This project demonstrates a modular data engineering approach, from raw data ingestion to model evaluation.

## 📁 Data Source
The dataset used in this project is publicly available on Kaggle:
- [Telecom Subscription Customer Churn Dataset](https://www.kaggle.com/datasets/beamhonor0911/telecom-subscription-customer-churn-dataset)
*Note: The raw data file is not uploaded to this repository for privacy and size considerations.*

## 🛠️ Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression)
- **Environment:** MacOS, VS Code

## 🔄 Pipeline Workflow
1. **Data Loading (`data_clean.py`):** Loads the raw CSV and handles initial data cleaning.
2. **Feature Engineering & Training (`model_train.py`):** Drops irrelevant columns, performs One-Hot Encoding on categorical features, handles missing values, and trains a Logistic Regression model.
3. **Pipeline Orchestration (`main.py`):** Connects data loading and model training into a single automated workflow.

## 📊 Results & Business Trade-off

The dataset has a severe class imbalance (~22% churn rate). A baseline model (DummyClassifier) was established, and `class_weight='balanced'` was applied to prioritize minority class detection.

| Model | Test Accuracy | ROC-AUC | PR-AUC (Primary Metric) | Churn Recall | Churn Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (Most Frequent) | 78.00% | 0.50 | ~0.22 | 0.00 | 0.00 |
| **Logistic Regression (Balanced)** | **0.71** | **0.77** | **0.33** | **0.68** | **0.21** |

*Note: A 5-fold Stratified Cross-Validation was performed to ensure model stability. The `classification_report` demonstrates the inherent trade-off: prioritizing high recall (68%) for the minority 'Churn' class significantly improves business value, even though overall accuracy drops.*

*Tools used: Python, Pandas, Scikit-learn, Git.*


## 🚀 How to Run
1. Clone the repository.
2. Download the dataset from the Kaggle link above and place it in the `data/` folder.
3. Run the pipeline:
   ```bash
   python3 main.py