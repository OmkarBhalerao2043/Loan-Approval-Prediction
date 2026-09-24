# src/compare_models.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import get_preprocessor

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC()
    }

    results = {}
    preprocessor = get_preprocessor()

    print("=" * 60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("=" * 60)

    for name, model in models.items():
        # Each model gets its own fresh pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        results[name] = {"Accuracy": accuracy, "F1": f1}

        print(f"{name:<25} | {accuracy:.4f}     | {f1:.4f}")

    return results