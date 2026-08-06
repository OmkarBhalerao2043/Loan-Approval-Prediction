from src.data_loader import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate_model
from src.compare_models import compare_models
from src.tune_model import tune_logistic_regression

def main():

    print("Loading Dataset...")

    df = load_data()

    print("Preprocessing Dataset...")

    X, y = preprocess(df)

    print("Training Random Forest...")

    model, X_train, X_test, y_train, y_test = train_model(X, y)

    print("Evaluating Random Forest...")

    evaluate_model(model, X_test, y_test)

    print("\nComparing Models...\n")

    compare_models(X_train, X_test, y_train, y_test)

    print("\nHyperparameter Tuning...\n")

    best_model = tune_logistic_regression(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()