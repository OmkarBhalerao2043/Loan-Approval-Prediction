import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from streamlit_shap import st_shap
from fpdf import FPDF
from sklearn.metrics import confusion_matrix

# Custom modules
from src.predict import predict
from src.explain import explain_prediction

# -----------------------------
# Page Config & CSS
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# -----------------------------
# PDF Generation Function
# -----------------------------
def generate_pdf_report(sample_data, status, probability):
    # Fix: Remove emojis for FPDF compatibility
    clean_status = status.replace("✅", "").replace("❌", "").strip()
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Loan Application Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)
    
    # Result
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Decision: {clean_status}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Approval Probability: {probability:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Applicant Details
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Applicant Details:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    
    for key, value in sample_data.items():
        pdf.cell(0, 8, f"{key}: {value}", new_x="LMARGIN", new_y="NEXT")
        
    # FIX: Convert bytearray to bytes for Streamlit compatibility
    return bytes(pdf.output())
# -----------------------------
# Sidebar: User Inputs
# -----------------------------
with st.sidebar:
    # Fallback text if image URL fails
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Loan Application")
    st.write("Adjust the parameters below to test the model.")
    
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

# -----------------------------
# Main Tabs
# -----------------------------
tab_pred, tab_exp, tab_info = st.tabs([
    "🚀 Prediction & Report", 
    "🧠 Explainability (SHAP)", 
    "📈 Model Performance"
])

# Process prediction if button is clicked, or if session state exists
if predict_btn:
    sample = {
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area
    }
    
    # Fetch backend data
    prediction, probability, scaled_df, original_df = predict(sample)
    explainer, shap_values = explain_prediction(scaled_df)
    
    approval_prob = probability[1] * 100
    status_text = "Approved ✅" if prediction == 1 else "Rejected ❌"
    
    # Save to session state so it persists across tabs
    st.session_state['results'] = {
        'sample': sample, 'prediction': prediction, 'approval_prob': approval_prob,
        'status_text': status_text, 'explainer': explainer, 'shap_values': shap_values,
        'scaled_df': scaled_df, 'original_df': original_df
    }

# =============================
# TAB 1: PREDICTION & REPORT
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
            st.write(f"- **Term:** {loan_term} months")
            
            # Generate and offer PDF download
            pdf_bytes = generate_pdf_report(res['sample'], res['status_text'], res['approval_prob'])
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_bytes,
                file_name="loan_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with col2:
            # Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['approval_prob'],
                title={'text': "Approval Probability", 'font': {'size': 24}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff4b4b"},  # Red
                        {'range': [40, 70], 'color': "#ffa500"},  # Orange
                        {'range': [70, 100], 'color': "#00cc96"}  # Green
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': res['approval_prob']
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("👈 Enter applicant details in the sidebar and click **Predict Loan Status**.")

# =============================
# TAB 2: EXPLAINABILITY (SHAP)
# =============================
with tab_exp:
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # Setup SHAP explanation object
        values = res['shap_values'][0, :, 1] if len(res['shap_values'].shape) == 3 else res['shap_values'][0]
        base_value = res['explainer'].expected_value[1] if isinstance(res['explainer'].expected_value, (list, np.ndarray)) else res['explainer'].expected_value
        
        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=res['scaled_df'][0], 
            feature_names=res['original_df'].columns.tolist()
        )
        
        st.subheader("Interactive Force Plot (Local)")
        st.write("Hover over the segments to see how each feature pushes the model's output from the base value to the final prediction.")
        # Render Interactive Force Plot
        st_shap(shap.force_plot(base_value, values, res['scaled_df'][0], feature_names=res['original_df'].columns.tolist()), height=150)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Waterfall Plot")
            st.write("Step-by-step breakdown of feature contributions.")
            fig_waterfall, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(explanation, max_display=8, show=False)
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)
            
        with col2:
            st.subheader("Global Feature Importance")
            st.write("Which features matter most across *all* applications? *(Simulated Background Data)*")
            
            # Generating a simulated global view for the dashboard visual
            dummy_shap = np.random.randn(100, len(res['original_df'].columns))
            dummy_data = np.random.randn(100, len(res['original_df'].columns))
            
            fig_summary, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(dummy_shap, dummy_data, feature_names=res['original_df'].columns.tolist(), show=False)
            st.pyplot(fig_summary)
            plt.close(fig_summary)
            
    else:
        st.info("👈 Generate a prediction first to see explanations.")

# =============================
# TAB 3: MODEL PERFORMANCE
# =============================
with tab_info:
    st.subheader("📊 Model Performance Dashboard")
    st.write("This tab monitors the historical performance of the Machine Learning model on the test set.")
    
    # Real metrics derived directly from your terminal output
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    metrics_col1.metric("Accuracy", "85.37%")
    metrics_col2.metric("Precision (Class 1)", "85.00%")
    metrics_col3.metric("Recall (Class 1)", "96.00%")
    metrics_col4.metric("F1-Score (Class 1)", "90.00%")
    metrics_col5.metric("ROC-AUC", "0.91") 
    
    st.divider()
    
    col_cm, col_info = st.columns([1, 1])
    
    with col_cm:
        st.write("### Confusion Matrix")
        # Real confusion matrix from your test output
        cm = np.array([[23, 15], [3, 82]]) 
        
        fig_cm, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rejected (0)', 'Approved (1)'], 
                    yticklabels=['Rejected (0)', 'Approved (1)'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
    with col_info:
        st.write("### Model Metadata")
        st.write("**Algorithm:** Random Forest Classifier")
        st.write("**Test Set Size:** 123 rows")
        st.write("**Last Trained:** August 6, 2026")
        st.write("**Best Logistic Regression Params (From Tuning):**")
        st.code("""
        {
            'C': 0.01, 
            'solver': 'liblinear'
        }
        """, language="json")