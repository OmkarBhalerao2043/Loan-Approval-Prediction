# src/predict.py
import pandas as pd
import joblib

from src.preprocess import engineer_features
import config

def predict(sample):
    # Load the unified pipeline
    pipeline = joblib.load(config.MODEL_PATH)
    
    df = pd.DataFrame([sample])

    # 1. Apply the exact same feature engineering used in training
    df = engineer_features(df)

    # 2. The pipeline handles all imputation, encoding, scaling, and prediction safely
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0]

    return prediction, probability, df