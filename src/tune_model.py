# src/tune_model.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

from src.preprocess import get_preprocessor

def tune_logistic_regression(X_train, X_test, y_train, y_test):

    # 1. Bundle preprocessing and modeling into a Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # 2. Append 'classifier__' to the parameters so GridSearchCV knows which step to tune
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    # 3. Pass the PIPELINE as the estimator, not just the model
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1 # Uses all CPU cores to run faster
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    prediction = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    print("=" * 60)
    print("Best Parameters")
    print(grid.best_params_)

    print("\nCross Validation Accuracy")
    print(f"{grid.best_score_:.4f}")

    print("\nTest Accuracy")
    print(f"{accuracy:.4f}")

    # Save it under a different name so it doesn't overwrite your primary Random Forest pipeline
    joblib.dump(best_model, "models/best_lr_model.pkl")
    
    return best_model