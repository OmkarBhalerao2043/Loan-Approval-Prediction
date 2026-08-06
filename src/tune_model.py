from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

def tune_logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    prediction = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("=" * 60)
    print("Best Parameters")
    print(grid.best_params_)

    print("\nCross Validation Accuracy")
    print(grid.best_score_)

    print("\nTest Accuracy")
    print(accuracy)

    joblib.dump(best_model, "models/best_model.pkl")
    
    return best_model