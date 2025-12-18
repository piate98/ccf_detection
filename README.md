# 💳 Credit Card Fraud Detection (Machine Learning)
* https://ccfdetection-985.streamlit.app/

This project builds and deploys a practical credit card fraud detection system using real transaction data.
The focus is on realistic feature engineering, imbalanced classification, probability calibration, and a production-ready Streamlit app.

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where:Fraudulent transactions are rare ,False positives are costly
Probability calibration matters more than raw accuracy

This project demonstrates:

* End-to-end ML workflow

* Robust preprocessing with mixed feature types

* Model calibration for realistic probability outputs

* Deployment without shipping large datasets

## 📊 Dataset

Source: Kaggle
Name: Credit Card Transactions Fraud Detection (No PCA)

🔗 Dataset link:
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Dataset characteristics

* ~555,000 transactions

* Strong class imbalance (~0.4% fraud)

*  Real-world categorical and numerical features

* No PCA anonymization (interpretable features)

⚠️ Note:
The dataset is not included in this repository due to GitHub file size limits.
It is used only during training and analysis, not required at inference time.

## 🧾 Features Used

The model learns from a combination of transaction, customer, and geographical features:

### Transaction features

* amt — transaction amount

* trans_hour — hour of transaction

* trans_dayofweek — weekday (0=Mon)

* trans_month — month

### Merchant & category

* merchant

* category

### Customer information

* gender

* age

* job

* state

* city_pop

### Location & geospatial

* lat, long — customer location

* merch_lat, merch_long — merchant location

### Target

* is_fraud (0 = legitimate, 1 = fraud)

## 🧠 Model & Approach
### Preprocessing

* ColumnTransformer

  * Numerical features → StandardScaler

  * Categorical features → OneHotEncoder(handle_unknown="ignore")

* Handling imbalance

  * class_weight="balanced"

  * Evaluation focused on recall, precision, ROC-AUC, not accuracy

### Models tested

* Logistic Regression (baseline)

* Random Forest

* Calibrated classifier (for realistic probabilities)

## Final choice

* Logistic Regression + calibration

Chosen for:

* Stable probabilities

* Interpretability

* Better real-world behavior than tree models

## 📈 Model Performance (Typical)
* Metric	Value
  * ROC-AUC	~0.95
* Fraud Recall	High (captures most fraud)
*  Precision	Low (expected for rare events)

In fraud detection, missing fraud is worse than flagging extra transactions.

🖥 Streamlit Application

The project includes a clean, production-ready Streamlit UI:

## App features

* Manual transaction input

* Fraud probability output

* Adjustable decision threshold

* No data leakage (merchant names cleaned)

## Why no dataset in the app?

* Large datasets should not ship with production apps

* Model inference requires only the trained pipeline

* Keeps deployment lightweight and reliable


Model deployment
* streamlit and github
