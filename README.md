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

## 📊 Results
The baseline Logistic Regression model achieved an accuracy of **90.06%** on the test set.

## 🚀 How to Run
1. Clone the repository.
2. Download the dataset from the Kaggle link above and place it in the `data/` folder.
3. Run the pipeline:
   ```bash
   python3 main.py