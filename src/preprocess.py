import pandas as pd


def preprocess(df):

    df = df.copy()

    # Drop unnecessary column
    df.drop(columns=["Loan_ID"], inplace=True)

    # Fill categorical missing values
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])

    # Fill numerical missing values
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X = pd.get_dummies(X, drop_first=True)

    import joblib
    joblib.dump(X.columns.tolist(), "models/features.pkl")

    y = y.map({"N": 0, "Y": 1})

    return X, y