from src.data_loader import load_data
from src.train import train_model

def main():
    print("Loading Dataset...")
    df = load_data()

    print("Training Final Production Model & Processing Pipeline...")
    # This now trains your tuned Logistic Regression and saves all UI artifacts
    model, X_train, X_test, y_train, y_test = train_model(df)

    print("Pipeline successfully trained and artifacts saved to models/ directory.")

if __name__ == "__main__":
    main()