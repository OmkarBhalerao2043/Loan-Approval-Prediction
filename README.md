# 🏦 AI Loan Approval Dashboard

An end-to-end Machine Learning application for predicting loan approval decisions using **Random Forest**, with **Explainable AI (SHAP)**, interactive dashboards, PDF reporting, and model performance monitoring.

---

## 🌐 Live Demo

Try the application here:

👉 **https://loan-approval-prediction-using-ai.streamlit.app/**

No installation required. Simply open the link, enter the applicant details, and explore the prediction, SHAP explainability, and model performance dashboard.

---

# Demo

<p align="center">

Loan Prediction → Explainability → Performance Dashboard

</p>

---

# Features

## Loan Prediction

- Predict loan approval instantly
- Random Forest classifier
- Approval/Rejection probability
- Interactive Plotly Gauge
- Professional UI built with Streamlit

---

## Explainable AI (SHAP)

Understand why the model predicted a decision.

Includes

- SHAP Force Plot
- SHAP Waterfall Plot
- Local Feature Importance
- Individual prediction explanations

This makes the prediction transparent and interview-ready.

---

## PDF Report Generation

Generate a professional report containing

- Applicant Information
- Loan Details
- Prediction
- Approval Probability

Download with one click.

---

## Model Performance Dashboard

Monitor model quality using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Includes

- Confusion Matrix
- Model Metadata
- Hyperparameter Information

---

# Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Feature Engineering

↓

One-Hot Encoding

↓

Train/Test Split

↓

Feature Scaling

↓

Random Forest Training

↓

Prediction API

↓

SHAP Explainability

↓

Streamlit Dashboard

---

# Feature Engineering

Three engineered features were created.

| Feature | Formula |
|----------|----------|
| TotalIncome | ApplicantIncome + CoapplicantIncome |
| EMI | LoanAmount / Loan_Amount_Term |
| Income_to_Loan_Ratio | TotalIncome / LoanAmount |

These significantly improve model understanding and predictive capability.

---

# Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **85.37%** |
| Precision | **85.00%** |
| Recall | **96.00%** |
| F1 Score | **90.00%** |
| ROC-AUC | **0.91** |

---

# Technologies Used

## Machine Learning

- Scikit-Learn
- Random Forest
- Logistic Regression
- GridSearchCV
- SHAP

## Data

- Pandas
- NumPy

## Visualization

- Plotly
- Matplotlib
- Seaborn

## Web Application

- Streamlit
- streamlit-shap

## Utilities

- Joblib
- FPDF2

---

# Project Structure

```text
Loan-Approval-Prediction
│
├── models
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── best_model.pkl
│
├── src
│   ├── compare_models.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── tune_model.py
│
├── app.py
├── main.py
├── style.css
├── requirements.txt
└── README.md
```

---

# Dashboard Overview

## Prediction Tab

✔ Applicant Information

✔ Financial Details

✔ Approval Gauge

✔ Prediction Status

✔ PDF Download

---

## Explainability Tab

✔ SHAP Force Plot

✔ SHAP Waterfall Plot

✔ Feature Contribution

✔ Local Explainability

---

## Performance Tab

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ Confusion Matrix

✔ Model Metadata

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Loan-Approval-Prediction.git
```

Move into project

```bash
cd Loan-Approval-Prediction
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

Train the model

```bash
python main.py
```

Launch application

```bash
streamlit run app.py
```

---

# Screenshots

Add screenshots here.

### Dashboard

```
images/dashboard.png
```

### Prediction

```
images/prediction.png
```

### SHAP Explainability

```
images/shap1.png
images/shap2.png
```

### Performance Dashboard

```
images/performance.png
```



# Author

**OMKAR BHALERAO**

GitHub

https://github.com/OmkarBhalerao2043

LinkedIn

https://www.linkedin.com/in/omkarbhalerao3

---

# License

MIT License