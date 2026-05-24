"""
Streamlit Dashboard for EthioHealth-AI
Interactive clinical decision support interface for Ethiopian EDs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import warnings

warnings.filterwarnings("ignore")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="EthioHealth-AI | ED Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-moderate { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# =========================
# Helper Functions
# =========================
def check_api_connection():
    try:
        r = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def make_prediction(patient_data):
    try:
        r = requests.post(
            f"{st.session_state.api_url}/predict",
            json=patient_data,
            timeout=10
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_model_info():
    try:
        r = requests.get(f"{st.session_state.api_url}/model-info", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def create_risk_indicator(level):
    if "HIGH" in level:
        return f"<span class='risk-high'>{level}</span>"
    elif "MODERATE" in level:
        return f"<span class='risk-moderate'>{level}</span>"
    return f"<span class='risk-low'>{level}</span>"


def simulate_ed_census():
    np.random.seed(42)
    n = np.random.randint(30, 60)

    return pd.DataFrame({
        "patient": [f"P{str(i).zfill(4)}" for i in range(n)],
        "predicted_los": np.random.exponential(8, n),
        "triage": np.random.choice([1,2,3,4,5], n, p=[0.05,0.15,0.35,0.30,0.15]),
        "chief_complaint": np.random.choice(
            ["Malaria","Trauma","Respiratory","Abdominal pain","Maternal","Other"], n
        ),
        "wait_time": np.random.uniform(0.5, 8, n)
    })


# =========================
# Sidebar Navigation (FIXED)
# =========================
def sidebar_menu():
    return st.sidebar.radio(
        "🧭 Navigation",
        [
            "🔮 Single Patient Prediction",
            "📊 ED Dashboard",
            "📈 Model Insights",
            "ℹ️ About"
        ]
    )


# =========================
# Main App
# =========================
def main():

    # Header
    col1, col2 = st.columns([3,1])

    with col1:
        st.markdown("# 🏥 EthioHealth-AI")
        st.markdown("**Localized Clinical Decision Support System**")
        st.markdown("Reducing ED overcrowding in Ethiopian referral hospitals")

    with col2:
        if check_api_connection():
            st.success("✓ Backend Connected")
        else:
            st.error("✗ Backend Not Connected")
            st.markdown("Run: `uvicorn api.main:app --reload`")

    st.divider()

    # ✅ FIX: page is now defined properly
    page = sidebar_menu()

    # =========================
    # PAGE 1
    # =========================
    if page == "🔮 Single Patient Prediction":

        st.header("Patient Triage & LOS Prediction")

        with st.form("patient_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 35)
                gender = st.selectbox("Gender", ["M","F"])
                region = st.selectbox("Region", ["Addis Ababa","Oromia","SNNP","Amhara"])
                chief = st.selectbox("Chief Complaint", ["Malaria","Trauma","Respiratory","Other"])

            with col2:
                bp_sys = st.number_input("Systolic BP", 70, 250, 130)
                hr = st.number_input("Heart Rate", 30, 200, 95)
                spo2 = st.number_input("SpO2", 60, 100, 96)
                temp = st.number_input("Temp", 35.0, 42.0, 37.0)

            triage = st.radio("Triage", [1,2,3,4,5])
            submit = st.form_submit_button("Predict")

        if submit:

            patient = {
                "age": age,
                "gender": gender,
                "region": region,
                "chief_complaint": chief,
                "systolic_bp": bp_sys,
                "heart_rate": hr,
                "spo2": spo2,
                "temperature_c": temp,
                "triage_category": triage
            }

            result = make_prediction(patient)

            if "error" not in result:
                st.success("Prediction Complete")

                c1,c2,c3 = st.columns(3)

                with c1:
                    st.metric("LOS", f"{result.get('predicted_los_hours','N/A')} hrs")

                with c2:
                    st.metric("Admission", f"{int(result.get('admission_probability',0)*100)}%")

                with c3:
                    st.markdown(create_risk_indicator(result.get("risk_level","LOW")), unsafe_allow_html=True)

                st.info(f"""
                **Recommendation:** {result.get('recommended_action','N/A')}
                """)

            else:
                st.error(result["error"])

    # =========================
    # PAGE 2
    # =========================
    elif page == "📊 ED Dashboard":

        st.header("ED Census Simulation")

        df = simulate_ed_census()

        st.metric("Patients", len(df))
        st.metric("Avg LOS", f"{df.predicted_los.mean():.1f}h")

        fig = go.Figure(data=[
            go.Histogram(x=df["predicted_los"])
        ])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df)

    # =========================
    # PAGE 3
    # =========================
    elif page == "📈 Model Insights":

        st.header("Model Insights")

        info = get_model_info()

        if info:
            st.write(info)
        else:
            st.warning("Backend not available")

    # =========================
    # PAGE 4
    # =========================
    elif page == "ℹ️ About":

        st.header("About EthioHealth-AI")

        st.markdown("""
        AI system for ED patient flow optimization in Ethiopian hospitals.
        """)


if __name__ == "__main__":
    main()