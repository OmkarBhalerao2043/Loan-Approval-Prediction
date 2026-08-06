import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_prediction(sample_df):

    model = joblib.load("models/model.pkl")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(sample_df)

    return explainer, shap_values