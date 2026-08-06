```markdown
# 🏦 AI-Powered Loan Approval Prediction & Explainability Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LIVE_URL_HERE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application designed to predict customer loan eligibility, provide transparent AI decision-making using **SHAP**, and deliver a comprehensive model performance dashboard.

---

## 🌟 Key Features

*   **Interactive Prediction Dashboard:** Input applicant information (demographics and financials) via a sidebar to instantly calculate approval/rejection probabilities using a real-time gauge chart.
*   **Local Explainability (SHAP):** Uses **SHAP Force Plots** and **Waterfall Plots** to break down exactly why an individual application was approved or denied.
*   **Automated PDF Reporting:** Instantly generate and download a professional summary report of the loan application and decision in PDF format.
*   **Model Performance Suite:** Interactive evaluation tab tracking model metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix).

---

## 📊 Model Performance & Tech Stack

*   **Core Algorithm:** Random Forest Classifier / Logistic Regression (Optimized via GridSearchCV)
*   **Evaluation Metrics:**
    *   **Accuracy:** 85.37%
    *   **Precision (Class 1):** 85.00%
    *   **Recall (Class 1):** 96.00%
    *   **F1-Score:** 90.00%
*   **Tech Stack:** 
    *   **Frontend/UI:** Streamlit, Plotly, Matplotlib, Seaborn
    *   **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
    *   **Explainability:** SHAP (SHapley Additive exPlanations)
    *   **Reporting:** FPDF2

---

## 📂 Project Structure

```text
Loan-Approval-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_model.pkl
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── notebooks/
│   └── data_understanding.ipynb
│
├── src/
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

## 🚀 How to Run Locally

Follow these steps to set up and run the project on your local machine:

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/Loan-Approval-Prediction.git](https://github.com/YOUR_USERNAME/Loan-Approval-Prediction.git)
cd Loan-Approval-Prediction

```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Pipeline (Optional)

To retrain the model and regenerate serialized pickles:

```bash
python main.py

```

### 5. Launch the Streamlit App

```bash
streamlit run app.py

```

---

## 🧠 Explainability Preview

The dashboard utilizes **SHAP** to ensure full transparency, complying with modern algorithmic fairness and explainability standards in FinTech.

* *Waterfall charts* detail additive feature attribution for individual applicants.
* *Feature importance breakdowns* isolate primary risk factors (e.g., Credit History, Loan Amount, Income-to-Loan Ratios).

---

## 👤 Author

**Your Name**

* [GitHub Profile](https://www.google.com/search?q=https://github.com/YOUR_USERNAME)
* [LinkedIn Profile](https://www.google.com/search?q=https://linkedin.com/in/YOUR_LINKEDIN)

```

```