import streamlit as st
import requests
from datetime import datetime
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="EthioHealth-AI", page_icon="🏥", layout="wide")

st.title("🏥 EthioHealth-AI Clinical Decision Support")

# ======================
# API CHECK
# ======================

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


if check_api():
    st.sidebar.success("✅ Backend Connected")
else:
    st.sidebar.error("❌ Backend Offline (start uvicorn first)")


# Navigation
page = st.sidebar.radio("Navigate", ["LOS Prediction", "Quick Diagnosis", "Full CDSS"])

# ======================
# LOS Prediction Page
# ======================
if page == "LOS Prediction":

    st.header("Length of Stay (LOS) Prediction")

    with st.form("patient_los"):

        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 1, 120, 30)
            gender = st.selectbox("Gender", ["M", "F"])
            complaint = st.selectbox("Complaint", ["Malaria", "Trauma", "Respiratory infection", "Typhoid", "Other"])

        with col2:
            systolic_bp = st.number_input("Systolic BP (mmHg)", 70, 250, 120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 30, 150, 80)
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 90)

        col3, col4 = st.columns(2)
        with col3:
            spo2 = st.number_input("SpO2 (%)", 60.0, 100.0, 95.0)
            temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
        
        with col4:
            triage = st.selectbox("Triage Category", [1, 2, 3, 4, 5])
            region = st.selectbox("Region", ["Addis Ababa", "Oromia", "SNNP", "Tigray"])

        submit_los = st.form_submit_button("Predict LOS")

    if submit_los:
        payload = {
            "age": age,
            "gender": gender,
            "chief_complaint": complaint,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "respiratory_rate": 20,
            "spo2": spo2,
            "temperature_c": temp,
            "triage_category": triage,
            "region": region,
            "transport_mode": "Walk",
            "imaging_ordered": "No",
            "lab_ordered": "No",
            "arrival_hour": 10
        }

        try:
            res = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()

            if "detail" in res:
                st.error(f"Error: {res['detail']}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted LOS", f"{res['predicted_los_hours']}h")
                with col2:
                    st.metric("Risk Level", res["risk_level"])
                with col3:
                    st.metric("Admission Probability", f"{res['admission_probability']*100:.0f}%")
                
                st.success(f"**Recommendation:** {res['recommended_action']}")
                st.info(f"LOS Category: {res['predicted_los_category']}")
                st.write(f"95% CI: {res['confidence_interval_lower']:.1f}h - {res['confidence_interval_upper']:.1f}h")

        except Exception as e:
            st.error(str(e))

# ======================
# Quick Diagnosis Page
# ======================
elif page == "Quick Diagnosis":

    st.header("Quick Symptom-based Diagnosis")

    symptoms_text = st.text_area("Enter symptoms (free text or comma-separated)", height=100, placeholder="e.g., fever, headache, chills")
    quick_btn = st.button("Quick Diagnose", key="quick_btn")

    if "quick_history" not in st.session_state:
        st.session_state["quick_history"] = []

    if quick_btn:
        if not symptoms_text or not symptoms_text.strip():
            st.warning("Please enter symptoms to diagnose.")
        else:
            if "," in symptoms_text:
                symptoms = [s.strip() for s in symptoms_text.split(",") if s.strip()]
            else:
                symptoms = [s.strip() for s in symptoms_text.split() if s.strip()]

            payload = {"symptoms": symptoms}

            try:
                r = requests.post(f"{API_URL}/diagnose", json=payload, timeout=5)
                res = r.json()

                candidates = res.get("candidates", [])

                if not candidates:
                    st.info("No likely diagnosis found. Consider full CDSS evaluation.")
                else:
                    for i, cand in enumerate(candidates[:3]):
                        with st.expander(f"{i+1}. {cand.get('disease')} (score: {cand.get('score')})"):
                            tests = cand.get('suggested_tests', [])
                            if tests:
                                st.write("🧪 Suggested tests:", ", ".join(tests))
                            tr = cand.get('suggested_treatment')
                            if tr:
                                st.write("💊 Suggested treatment:", tr)

                entry = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "symptoms": symptoms,
                    "candidates": candidates
                }
                st.session_state["quick_history"].insert(0, entry)

            except Exception as e:
                st.error(str(e))

    if st.session_state["quick_history"]:
        st.subheader("Recent Diagnoses")
        for h in st.session_state["quick_history"][:10]:
            st.caption(f"{h['time']} — {', '.join(h['symptoms'][:3])}")

# ======================
# Full CDSS Page
# ======================
else:

    st.header("🏥 Comprehensive Clinical Decision Support System")
    
    st.info(
        "⚠️ This tool assists licensed healthcare professionals. Final diagnostic and treatment decisions "
        "must be made by qualified physicians. Always verify outputs against clinical judgment."
    )

    with st.form("cdss_full"):
        
        # DEMOGRAPHICS
        st.subheader("1. Patient Demographics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age_cdss = st.number_input("Age (years)", 1, 120, 35, key="age_cdss")
        with col2:
            gender_cdss = st.selectbox("Gender", ["M", "F"], key="gender_cdss")
        with col3:
            weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
        with col4:
            height = st.number_input("Height (cm)", 100, 220, 170)

        # SYMPTOMS
        st.subheader("2. Chief Complaint & Symptoms")
        symptoms_input = st.text_area(
            "Enter all symptoms (comma-separated or free text)",
            height=80,
            placeholder="fever, headache, body ache, chills"
        )
        col1, col2 = st.columns(2)
        with col1:
            duration = st.number_input("Duration (days)", 0.0, 365.0, 3.0)
        with col2:
            severity_cdss = st.selectbox("Severity", ["mild", "moderate", "severe"])

        # VITAL SIGNS
        st.subheader("3. Vital Signs")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            temp_cdss = st.number_input("Temp (°C)", 35.0, 42.0, 38.5)
        with col2:
            sys_bp = st.number_input("SysBP", 70, 250, 120)
        with col3:
            dia_bp = st.number_input("DiaBP", 30, 150, 80)
        with col4:
            hr_cdss = st.number_input("HR (bpm)", 30, 200, 95)
        with col5:
            spo2_cdss = st.number_input("SpO2 (%)", 50.0, 100.0, 96.0)

        col6 = st.columns(1)[0]
        with col6:
            rr_cdss = st.number_input("RR (breaths/min)", 8, 60, 18)

        # MEDICAL HISTORY
        st.subheader("4. Medical History")
        col1, col2 = st.columns(2)
        with col1:
            chronic = st.multiselect(
                "Chronic diseases",
                ["Diabetes", "Hypertension", "HIV/AIDS", "Asthma", "TB", "None"]
            )
        with col2:
            meds_input = st.text_area("Current medications (comma-separated)", height=60)

        allergies_input = st.text_area("Drug allergies (comma-separated)", height=60, placeholder="e.g., Penicillin, Sulfa")

        # LAB & IMAGING
        st.subheader("5. Lab Results & Imaging (if available)")
        imaging_input = st.text_area("Imaging findings", height=60, placeholder="e.g., Chest X-ray: pneumonia infiltrate")

        submit_cdss = st.form_submit_button("🔍 Run Full CDSS Analysis")

    if submit_cdss:
        if not symptoms_input.strip():
            st.warning("Please enter symptoms.")
        else:
            # Parse symptoms
            if "," in symptoms_input:
                syms = [s.strip() for s in symptoms_input.split(",") if s.strip()]
            else:
                syms = symptoms_input.split()

            # Parse meds
            meds = [m.strip() for m in meds_input.split(",")] if meds_input else []

            # Parse allergies
            allg = [a.strip() for a in allergies_input.split(",")] if allergies_input else []

            # Parse imaging
            img_list = [i.strip() for i in imaging_input.split(",")] if imaging_input else []

            payload = {
                "age": age_cdss,
                "gender": gender_cdss,
                "weight_kg": weight,
                "height_cm": height,
                "symptoms": syms,
                "duration_days": duration,
                "severity": severity_cdss,
                "temperature_c": temp_cdss,
                "systolic_bp": sys_bp,
                "diastolic_bp": dia_bp,
                "heart_rate": hr_cdss,
                "respiratory_rate": rr_cdss,
                "spo2": spo2_cdss,
                "chronic_diseases": chronic,
                "medications": meds,
                "allergies": allg,
                "imaging_findings": img_list
            }

            try:
                r = requests.post(f"{API_URL}/cdss", json=payload, timeout=10)
                cdss_result = r.json()

                # Display results
                st.subheader("📊 CDSS Analysis Result")

                # Risk assessment banner
                risk = cdss_result.get("risk_assessment", {})
                if risk.get("emergency_risk") == "HIGH":
                    st.error(f"⚠️ **EMERGENCY RISK: {risk.get('emergency_risk')}** | Referral: {cdss_result.get('recommended_referral', 'N/A')}")
                else:
                    st.info(f"Risk Level: {risk.get('emergency_risk')} | Referral: {cdss_result.get('recommended_referral', 'Standard')}")

                # Differential diagnoses
                st.subheader("🔬 Possible Diagnoses")
                diags = cdss_result.get("possible_diagnoses", [])
                for i, diag in enumerate(diags[:5]):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{i+1}. {diag.get('disease')}**")
                    with col2:
                        severity_badge = "🔴" if diag.get("severity") == "HIGH" else ("🟡" if diag.get("severity") == "MODERATE" else "🟢")
                        st.write(f"{severity_badge} {diag.get('severity')}")
                    with col3:
                        st.write(f"**{diag.get('probability')}%**")
                    
                    with st.expander(f"Details for {diag.get('disease')}"):
                        st.write("**Supporting Evidence:**")
                        for ev in diag.get("supporting_evidence", []):
                            st.write(f"  • {ev}")
                        if diag.get("red_flags_detected"):
                            st.error(f"🚨 **Red flags:** {', '.join(diag['red_flags_detected'])}")
                        st.write("**Recommended Tests:**", ", ".join(diag.get("recommended_tests", [])))

                # Recommendations
                st.subheader("💊 Recommended Tests & Management")
                tests = cdss_result.get("recommended_tests", [])
                if tests:
                    st.write("**Tests to order:**")
                    for test in tests:
                        st.write(f"  • {test}")

                st.subheader("⚠️ Clinical Warnings")
                st.write(cdss_result.get("disclaimer"))

                # Confidence
                st.caption(f"Diagnosis Confidence: {cdss_result.get('confidence_score', 0)}%")

            except Exception as e:
                st.error(f"API Error: {str(e)}")

