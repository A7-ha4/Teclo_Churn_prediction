# 📊 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on their account activity and service usage.

---

## 🧠 Project Overview

Customer churn is a major issue in the telecom industry. This project builds a classification model using machine learning to predict churn and help businesses retain valuable customers.

---

## 📁 Dataset

- **Source:** [IBM Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 rows, 21 columns
- **Target:** `Churn` (Yes/No)

---

## 🚀 Technologies Used

- Python 3
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (`SMOTE`)
- Optuna (Hyperparameter optimization)
- Joblib (Model saving)
- Category Encoders

---

## ⚙️ Workflow

1. **Data Cleaning**
   - Handled missing values (especially in `TotalCharges`)
   - Dropped irrelevant columns like `customerID`
   - Converted categorical variables

2. **Feature Engineering**
   - Encoding with `TargetEncoder` and `BinaryEncoder`
   - Feature selection using `SelectFromModel`
   - **Standardization using `StandardScaler`**

3. **Class Imbalance**
   - Solved using `SMOTE` to balance `Churn` classes

4. **Modeling**
   - Combined 3 models using `VotingClassifier`:
     - Logistic Regression
     - Random Forest
     - XGBoost
   - Used `Optuna` for tuning model parameters

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix, Classification Report

6. **Model Saving**
   - Final model saved as `trained_model.pkl` for deployment

---

## 📈 Results

- High-performing voting ensemble model
- Balanced precision and recall
- Improved generalization with Optuna tuning

---

## 🛠️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/A7-ha4/Teclo_Churn_prediction.git
   cd Teclo_Churn_prediction
