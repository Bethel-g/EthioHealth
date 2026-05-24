import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="EthioHealth-AI", page_icon="🏥")

st.title("🏥 EthioHealth-AI Dashboard")

# ======================
# API CHECK
# ======================
def check_api():
    try:
        r = requests.get(f"{API_URL}/health")
        return r.status_code == 200
    except:
        return False


if check_api():
    st.success("Backend Connected ✅")
else:
    st.error("Backend NOT Running ❌ (run uvicorn first)")


# ======================
# INPUT FORM
# ======================
with st.form("patient"):

    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender", ["M", "F"])
    complaint = st.selectbox("Complaint", ["Malaria", "Trauma", "Other"])

    systolic_bp = st.number_input("Systolic BP", 70, 250, 120)
    diastolic_bp = st.number_input("Diastolic BP", 30, 150, 80)

    heart_rate = st.number_input("Heart Rate", 30, 200, 90)
    spo2 = st.number_input("SpO2", 60, 100, 95)
    temp = st.number_input("Temperature", 35.0, 42.0, 37.0)

    triage = st.selectbox("Triage", [1, 2, 3, 4, 5])

    submit = st.form_submit_button("Predict LOS")


if submit:

    payload = {
        "age": age,
        "gender": gender,
        "chief_complaint": complaint,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temperature_c": temp,
        "triage_category": triage,
        "region": "Addis Ababa"
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload).json()

        if "detail" in res:
            st.error(res["detail"])
        else:
            st.metric("LOS (hours)", res["predicted_los_hours"])
            st.metric("Risk Level", res["risk_level"])
            st.write(res["recommended_action"])

    except Exception as e:
        st.error(str(e))