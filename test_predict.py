from src.predict import predict
from src.explain import explain_prediction

sample = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}

prediction, probability, scaled_df, original_df = predict(sample)

explainer, shap_values = explain_prediction(scaled_df)

print(type(shap_values))
print(len(shap_values) if hasattr(shap_values, "__len__") else "No length")
print(shap_values)