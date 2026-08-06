import pandas as pd

def load_data():

    df = pd.read_csv("hf://datasets/mariosyahirhalimm/loan_prediction_dataset/loan_prediction_dataset.csv")

    return df