# Add these imports at the top of src/train.py
import json
from datetime import datetime
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Inside your train_model function, AFTER pipeline.fit()...

def train_model(df):
    # ... (Feature engineering, splitting, and pipeline.fit code remains the same) ...

    # 1. Evaluate on test set to generate real metrics
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "last_trained": datetime.now().strftime("%B %d, %Y"),
        "test_set_size": len(X_test),
        "algorithm": "Random Forest Classifier",
        "accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}%",
        "precision": f"{precision_score(y_test, y_pred) * 100:.2f}%",
        "recall": f"{recall_score(y_test, y_pred) * 100:.2f}%",
        "f1_score": f"{f1_score(y_test, y_pred) * 100:.2f}%",
        "roc_auc": f"{roc_auc_score(y_test, y_prob):.2f}"
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

    # 2. Generate and save Global SHAP values
    rf_model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Transform a sample of training data for the global explainer (using 100 rows to keep file size small)
    X_train_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    X_train_transformed = preprocessor.transform(X_train_sample)
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values_global = explainer.shap_values(X_train_transformed)
    
    # Get feature names after preprocessing
    feature_names = (
        preprocessor.named_transformers_["num"].get_feature_names_out().tolist() +
        preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
    )

    joblib.dump(shap_values_global, "models/shap_values_global.pkl")
    joblib.dump(X_train_transformed, "models/X_train_transformed.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")

    joblib.dump(pipeline, "models/pipeline.pkl")

    return pipeline, X_train, X_test, y_train, y_test