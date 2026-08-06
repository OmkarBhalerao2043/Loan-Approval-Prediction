import joblib
import pandas as pd


def predict(sample):

    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")

    df = pd.DataFrame([sample])

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["TotalIncome"] = (
        df["ApplicantIncome"] +
        df["CoapplicantIncome"]
    )

    df["EMI"] = (
        df["LoanAmount"] /
        df["Loan_Amount_Term"]
    )

    df["Income_to_Loan_Ratio"] = (
        df["TotalIncome"] /
        df["LoanAmount"]
    )

    df = pd.get_dummies(df)

    df = df.reindex(columns=features, fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]

    probability = model.predict_proba(df_scaled)[0]

    return prediction, probability, df_scaled, df