import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from streamlit_shap import st_shap
from fpdf import FPDF
import joblib
import json

# Custom modules
from src.predict import predict
from src.explain import explain_prediction

# -----------------------------
# Caching Application Artifacts
# -----------------------------
@st.cache_resource
def load_system_artifacts():
    """Loads all models and metadata into RAM once to prevent disk I/O bottlenecks."""
    try:
        pipeline = joblib.load("models/pipeline.pkl")
        # Load SHAP and Feature Data
        shap_values_global = joblib.load("models/shap_values_global.pkl")
        X_train_transformed = joblib.load("models/X_train_transformed.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        
        # Load Metrics
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
            
        return pipeline, shap_values_global, X_train_transformed, feature_names, metrics
    except FileNotFoundError:
        st.error("System artifacts not found. Please run the training script first.")
        st.stop()

# Load everything at startup
pipeline, global_shap, global_data, feature_names, model_metrics = load_system_artifacts()

# -----------------------------
# Page Config & CSS
# -----------------------------
st.set_page_config(page_title="Loan Approval Dashboard", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# (PDF Generation function remains exactly the same as your code)
def generate_pdf_report(sample_data, status, probability):
    clean_status = status.replace("✅", "").replace("❌", "").strip()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Loan Application Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Decision: {clean_status}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Approval Probability: {probability:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Applicant Details:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    for key, value in sample_data.items():
        pdf.cell(0, 8, f"{key}: {value}", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Loan Application")
    
    st.subheader("👤 Applicant Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.subheader("💰 Financial Info")
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=150.0)
    loan_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
    credit_history = st.selectbox("Credit History", [1, 0])

    predict_btn = st.button("🚀 Predict Loan Status", use_container_width=True)

# -----------------------------
# Main Dashboard Header
# -----------------------------
st.title("🏦 AI Loan Approval Dashboard")
st.markdown("Predict applicant eligibility, explain AI decisions, and monitor model performance.")
st.divider()

tab_pred, tab_exp, tab_info = st.tabs(["🚀 Prediction & Report", "🧠 Explainability (SHAP)", "📈 Model Performance"])

# -----------------------------
# Inference Trigger (With Safety Net)
# -----------------------------
if predict_btn:
    try:
        sample = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history, "Property_Area": property_area
        }
        
        # 1. Unpack the values that predict.py returns
        prediction, probability, processed_df = predict(sample)
        
        # 2. Extract the preprocessor and model from our cached pipeline
        preprocessor = pipeline.named_steps["preprocessor"]
        
        # Renamed to lr_model to reflect our Logistic Regression shift
        lr_model = pipeline.named_steps["classifier"] 
        
        # 3. Transform the data and generate SHAP values in memory
        transformed_data = preprocessor.transform(processed_df)
        
        # CRITICAL FIX: Use LinearExplainer and pass the cached global_data background!
        explainer = shap.LinearExplainer(lr_model, global_data)
        shap_values = explainer.shap_values(transformed_data)
        
        approval_prob = probability[1] * 100
        status_text = "Approved ✅" if prediction == 1 else "Rejected ❌"
        
        # 4. Save to session state
        st.session_state['results'] = {
            'sample': sample, 'prediction': prediction, 'approval_prob': approval_prob,
            'status_text': status_text, 'explainer': explainer, 'shap_values': shap_values,
            'scaled_df': transformed_data, 
            'original_df': processed_df
        }
    except Exception as e:
        st.error(f"An error occurred while processing the application: {str(e)}")
             
# =============================
# TAB 1: PREDICTION & REPORT (Unchanged)
# =============================
with tab_pred:
    if 'results' in st.session_state:
        res = st.session_state['results']
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Application Status")
            if res['prediction'] == 1:
                st.success(f"## {res['status_text']}")
            else:
                st.error(f"## {res['status_text']}")
                
            st.write("### Quick Summary")
            st.write(f"- **Total Income:** ${applicant_income + coapplicant_income:,.2f}")
            st.write(f"- **Loan Amount:** ${loan_amount:,.2f}")
            
            pdf_bytes = generate_pdf_report(res['sample'], res['status_text'], res['approval_prob'])
            st.download_button(label="📄 Download Prediction Report", data=pdf_bytes, file_name="loan_report.pdf", mime="application/pdf", use_container_width=True)
            
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=res['approval_prob'],
                title={'text': "Approval Probability", 'font': {'size': 24}}, number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 70], 'color': "#ffa500"}, {'range': [70, 100], 'color': "#00cc96"}],
                       'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': res['approval_prob']}}
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Enter applicant details in the sidebar and click **Predict Loan Status**.")

# =============================
# TAB 2: EXPLAINABILITY (Using Real Data)
# =============================
with tab_exp:
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        values = res['shap_values'][0, :, 1] if len(res['shap_values'].shape) == 3 else res['shap_values'][0]
        base_value = res['explainer'].expected_value[1] if isinstance(res['explainer'].expected_value, (list, np.ndarray)) else res['explainer'].expected_value
        
        explanation = shap.Explanation(values=values, base_values=base_value, data=res['scaled_df'][0], feature_names=feature_names)
        
        st.subheader("Interactive Force Plot (Local)")
        st_shap(shap.force_plot(base_value, values, res['scaled_df'][0], feature_names=feature_names), height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Waterfall Plot")
            fig_waterfall, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(explanation, max_display=8, show=False)
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)
            
        with col2:
            st.subheader("Global Feature Importance")
            st.write("Based on actual training distribution.")
            # FIX: Replaced dummy data with actual cached SHAP values
            fig_summary, ax = plt.subplots(figsize=(8, 4))
            
            # Account for binary classification shape differences in SHAP arrays
            global_vals_to_plot = global_shap[:, :, 1] if len(global_shap.shape) == 3 else global_shap
            
            shap.summary_plot(global_vals_to_plot, global_data, feature_names=feature_names, show=False)
            st.pyplot(fig_summary)
            plt.close(fig_summary)
    else:
        st.info("👈 Generate a prediction first to see explanations.")

# =============================
# TAB 3: MODEL PERFORMANCE (Dynamic)
# =============================
with tab_info:
    st.subheader("📊 Model Performance Dashboard")
    
    # Dynamically loading from metrics.json
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    metrics_col1.metric("Accuracy", model_metrics.get("accuracy", "N/A"))
    metrics_col2.metric("Precision", model_metrics.get("precision", "N/A"))
    metrics_col3.metric("Recall", model_metrics.get("recall", "N/A"))
    metrics_col4.metric("F1-Score", model_metrics.get("f1_score", "N/A"))
    metrics_col5.metric("ROC-AUC", model_metrics.get("roc_auc", "N/A")) 
    
    st.divider()
    
    st.write("### Model Metadata")
    st.write(f"**Algorithm:** {model_metrics.get('algorithm', 'Unknown')}")
    st.write(f"**Test Set Size:** {model_metrics.get('test_set_size', 'Unknown')} rows")
    st.write(f"**Last Trained:** {model_metrics.get('last_trained', 'Unknown')}")