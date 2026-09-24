# src/preprocess.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def engineer_features(df):
    """Applies custom feature engineering identically across train and serve."""
    df = df.copy()
    
    # Adding a small constant (1e-5) prevents division by zero errors
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1e-5)
    df["Income_to_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] + 1e-5)
    
    return df

def get_preprocessor():
    """Builds a robust, leak-proof preprocessing pipeline."""
    num_features = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "TotalIncome",
        "EMI", "Income_to_Loan_Ratio"
    ]
    
    cat_features = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area"
    ]

    # Impute missing numericals with the median, then scale
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Impute missing categoricals with the mode, then one-hot encode
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Bundle them together
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])
    
    return preprocessor