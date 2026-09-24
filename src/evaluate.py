# src/evaluate.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def evaluate_model(pipeline, X_test, y_test):
    prediction = pipeline.predict(X_test)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    print(f"Accuracy : {accuracy_score(y_test, prediction):.4f}")
    print(f"F1-Score : {f1_score(y_test, prediction):.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))
    print("\nClassification Report:")
    print(classification_report(y_test, prediction))