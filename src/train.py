# src/train.py
import json
from datetime import datetime
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

from src.preprocess import engineer_features, get_preprocessor

def train_model(df):
    # 1. Engineer features before splitting
    df = engineer_features(df)

    # 2. Separate features and target
    X = df.drop(columns=["Loan_ID", "Loan_Status"])
    y = df["Loan_Status"].map({"N": 0, "Y": 1})

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Assemble and fit the pipeline with the TUNED LOGISTIC REGRESSION
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(C=0.01, solver="liblinear", max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the production model
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate real confusion matrix and convert to list for JSON
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "last_trained": datetime.now().strftime("%B %d, %Y"),
        "test_set_size": len(X_test),
        "algorithm": "Tuned Logistic Regression (C=0.01, liblinear)",
        "accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}%",
        "precision": f"{precision_score(y_test, y_pred) * 100:.2f}%",
        "recall": f"{recall_score(y_test, y_pred) * 100:.2f}%",
        "f1_score": f"{f1_score(y_test, y_pred) * 100:.2f}%",
        "roc_auc": f"{roc_auc_score(y_test, y_prob):.2f}",
        "confusion_matrix": cm 
    }

    # Save metrics.json
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    # 6. Generate Global SHAP values
    lr_model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    X_train_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    X_train_transformed = preprocessor.transform(X_train_sample)
    
    # CRITICAL FIX: Use LinearExplainer for Logistic Regression
    explainer = shap.LinearExplainer(lr_model, X_train_transformed)
    shap_values_global = explainer.shap_values(X_train_transformed)
    
    feature_names = (
        preprocessor.named_transformers_["num"].get_feature_names_out().tolist() +
        preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    )

    # 7. Save ALL required artifacts
    joblib.dump(shap_values_global, "models/shap_values_global.pkl")
    joblib.dump(X_train_transformed, "models/X_train_transformed.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(pipeline, "models/pipeline.pkl")

    return pipeline, X_train, X_test, y_train, y_test