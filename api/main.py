# """
# FastAPI Backend for EthioHealth-AI
# Provides REST API endpoints for LOS prediction and clinical decision support
# """

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import Optional, List
# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# import warnings

# warnings.filterwarnings('ignore')

# # Initialize FastAPI app
# app = FastAPI(
#     title="EthioHealth-AI API",
#     description="Clinical Decision Support System for Ethiopian ED",
#     version="1.0.0"
# )

# # Load model and preprocessor
# MODEL_DIR = Path(__file__).parent.parent / "models"
# PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"
# MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"

# # Global variables
# preprocessor = None
# model = None


# @app.on_event("startup")
# async def load_model():
#     """Load model and preprocessor on startup."""
#     global preprocessor, model

#     try:
#         if PREPROCESSOR_FILE.exists():
#             preprocessor = joblib.load(PREPROCESSOR_FILE)
#         if MODEL_FILE.exists():
#             model = joblib.load(MODEL_FILE)
#             print("✓ Model loaded successfully")
#         else:
#             print("⚠ Model not found. Train model first.")
#     except Exception as e:
#         print(f"✗ Error loading model: {e}")


# # ===== Pydantic Models (Request/Response Schema) =====

# class PatientInput(BaseModel):
#     """Patient input for LOS prediction."""
#     age: float = Field(..., gt=0, le=120, description="Age in years")
#     gender: str = Field(..., description="M or F")
#     chief_complaint: str = Field(..., description="Main presenting complaint")
#     systolic_bp: float = Field(..., ge=70, le=250, description="Systolic BP in mmHg")
#     diastolic_bp: float = Field(..., ge=30, le=150, description="Diastolic BP in mmHg")
#     heart_rate: float = Field(..., ge=30, le=200, description="Heart rate in bpm")
#     respiratory_rate: float = Field(..., ge=8, le=60, description="Respiratory rate in breaths/min")
#     spo2: float = Field(..., ge=60, le=100, description="Oxygen saturation percentage")
#     temperature_c: float = Field(..., ge=35, le=42, description="Temperature in Celsius")
#     triage_category: int = Field(..., ge=1, le=5, description="1=Critical, 5=Non-urgent")
#     region: str = Field(..., description="Patient region")
#     transport_mode: str = Field(..., description="Walk, Car, Taxi, or Ambulance")
#     imaging_ordered: str = Field(default="No", description="No, X-ray, Ultrasound, CT")
#     lab_ordered: str = Field(default="No", description="No, Basic, Extended")
#     arrival_hour: int = Field(..., ge=0, le=23, description="Hour of arrival")


# class LOSPrediction(BaseModel):
#     """LOS prediction response."""
#     predicted_los_hours: float = Field(..., description="Predicted Length of Stay in hours")
#     predicted_los_category: str = Field(..., description="Fast-track, Standard, or Extended")
#     confidence_interval_lower: float = Field(..., description="95% CI lower bound")
#     confidence_interval_upper: float = Field(..., description="95% CI upper bound")
#     admission_probability: float = Field(..., description="Probability of admission (0-1)")
#     recommended_action: str = Field(..., description="Clinical recommendation")
#     risk_level: str = Field(..., description="🟢 LOW, 🟡 MODERATE, 🔴 HIGH")


# class HealthCheck(BaseModel):
#     """Health check response."""
#     status: str = Field(..., description="Server status")
#     model_loaded: bool = Field(..., description="Whether model is loaded")
#     version: str = Field(..., description="API version")


# # ===== Helper Functions =====

# def preprocess_input(patient: PatientInput) -> np.ndarray:
#     """
#     Preprocess patient input for model prediction.

#     Args:
#         patient: PatientInput instance

#     Returns:
#         Preprocessed feature vector (1, n_features)
#     """
#     # Create feature dataframe matching training format
#     data = {
#         "age": [patient.age],
#         "gender": [patient.gender],
#         "chief_complaint": [patient.chief_complaint],
#         "systolic_bp": [patient.systolic_bp],
#         "diastolic_bp": [patient.diastolic_bp],
#         "heart_rate": [patient.heart_rate],
#         "respiratory_rate": [patient.respiratory_rate],
#         "spo2": [patient.spo2],
#         "temperature_c": [patient.temperature_c],
#         "triage_category": [patient.triage_category],
#         "region": [patient.region],
#         "transport_mode": [patient.transport_mode],
#         "imaging_ordered": [patient.imaging_ordered],
#         "lab_ordered": [patient.lab_ordered],
#         "arrival_hour": [patient.arrival_hour],
#         "is_peak_hour": [1 if (patient.arrival_hour >= 8 and patient.arrival_hour <= 11) or
#                              (patient.arrival_hour >= 18 and patient.arrival_hour <= 22) else 0],
#         "is_weekend": [1 if pd.Timestamp.now().weekday() in [5, 6] else 0],
#         "is_night_shift": [1 if patient.arrival_hour in range(22, 6) else 0],
#         "complaint_risk_score": [{"Malaria": 3, "Trauma": 4, "Maternal emergency": 4,
#                                   "Typhoid": 3, "Respiratory infection": 2, "Abdominal pain": 2,
#                                   "Hypertensive emergency": 2, "Diarrheal disease": 1,
#                                   "Other": 1}.get(patient.chief_complaint, 1)],
#         "transport_risk": [{"Walk": 1, "Taxi": 2, "Car": 2, "Ambulance": 4}.get(patient.transport_mode, 1)],
#         "resource_intensity": [
#             int(patient.imaging_ordered != "No") +
#             int(patient.lab_ordered != "No") +
#             (1 if patient.triage_category <= 2 else 0)
#         ]
#     }

#     df = pd.DataFrame(data)

#     # Apply preprocessing
#     X_processed, _ = preprocessor.prepare_dataset(df, fit=False)

#     return X_processed.values


# def classify_los(predicted_hours: float) -> tuple:
#     """
#     Classify predicted LOS into clinical categories.

#     Args:
#         predicted_hours: Predicted LOS in hours

#     Returns:
#         (category, risk_level, recommended_action)
#     """
#     if predicted_hours < 4:
#         return ("Fast-track (< 4h)", "🟢 LOW", "Can wait for routine physician review")
#     elif predicted_hours < 12:
#         return ("Standard (4-12h)", "🟡 MODERATE", "Escalate to physician. Admit likely.")
#     else:
#         return ("Extended (> 12h)", "🔴 HIGH", "URGENT: Escalate immediately. ICU prep.")


# def estimate_admission_probability(predicted_hours: float, triage_cat: int) -> float:
#     """
#     Estimate probability of admission based on predicted LOS and triage.

#     Args:
#         predicted_hours: Predicted LOS
#         triage_cat: Triage category (1-5)

#     Returns:
#         Probability of admission (0-1)
#     """
#     # Empirical mapping: longer LOS + higher acuity = higher admission prob
#     base_prob = {1: 0.95, 2: 0.80, 3: 0.50, 4: 0.30, 5: 0.15}.get(triage_cat, 0.5)

#     # Adjust by LOS
#     if predicted_hours > 12:
#         admission_prob = min(base_prob + 0.20, 1.0)
#     elif predicted_hours > 8:
#         admission_prob = base_prob
#     else:
#         admission_prob = max(base_prob - 0.15, 0.0)

#     return admission_prob


# # ===== API Endpoints =====

# @app.get("/", tags=["Info"])
# async def root():
#     """Root endpoint."""
#     return {
#         "message": "EthioHealth-AI Clinical Decision Support API",
#         "docs": "/docs",
#         "health": "/health"
#     }


# @app.get("/health", response_model=HealthCheck, tags=["Health"])
# async def health_check():
#     """Health check endpoint."""
#     return HealthCheck(
#         status="✓ Running",
#         model_loaded=model is not None,
#         version="1.0.0"
#     )


# @app.post("/predict", response_model=LOSPrediction, tags=["Prediction"])
# async def predict_los(patient: PatientInput):
#     """
#     Predict Length of Stay (LOS) for a patient.

#     This endpoint provides clinical decision support for ED resource planning:
#     - Predicts patient LOS in hours
#     - Estimates admission probability
#     - Provides risk level and recommendations

#     **Clinical Note**: This is a decision support tool. Final decisions should
#     be made by qualified healthcare professionals.
#     """
#     if model is None or preprocessor is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded. Please train and save model first."
#         )

#     try:
#         # Preprocess input
#         X_processed = preprocess_input(patient)

#         # Make prediction
#         predicted_los = float(model.predict(X_processed)[0])

#         # Ensure prediction is positive
#         predicted_los = max(predicted_los, 0.5)

#         # Classify and generate recommendations
#         category, risk_level, action = classify_los(predicted_los)

#         # Estimate admission probability
#         admission_prob = estimate_admission_probability(predicted_los, patient.triage_category)

#         # Calculate 95% confidence interval (approximately ±1.96 * RMSE)
#         rmse_estimate = 1.5  # From model evaluation
#         ci_lower = max(predicted_los - 1.96 * rmse_estimate, 0.5)
#         ci_upper = predicted_los + 1.96 * rmse_estimate

#         return LOSPrediction(
#             predicted_los_hours=round(predicted_los, 2),
#             predicted_los_category=category,
#             confidence_interval_lower=round(ci_lower, 2),
#             confidence_interval_upper=round(ci_upper, 2),
#             admission_probability=round(admission_prob, 2),
#             recommended_action=action,
#             risk_level=risk_level
#         )

#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Prediction error: {str(e)}"
#         )


# @app.post("/batch-predict", tags=["Prediction"])
# async def batch_predict(patients: List[PatientInput]):
#     """
#     Predict LOS for multiple patients (batch prediction).

#     Useful for ED overview dashboards showing predicted LOS for all current patients.
#     """
#     if model is None or preprocessor is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not loaded. Please train and save model first."
#         )

#     predictions = []
#     for patient in patients:
#         try:
#             pred = await predict_los(patient)
#             predictions.append({
#                 "patient": patient.dict(),
#                 "prediction": pred.dict()
#             })
#         except Exception as e:
#             predictions.append({
#                 "patient": patient.dict(),
#                 "error": str(e)
#             })

#     return {"predictions": predictions, "count": len(predictions)}


# @app.get("/model-info", tags=["Info"])
# async def model_info():
#     """Get information about the trained model."""
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     return {
#         "model_type": type(model).__name__,
#         "expected_performance": {
#             "MAE": "±1.5 hours",
#             "RMSE": "2.1 hours",
#             "R²": "0.82",
#             "MAPE": "18%"
#         },
#         "clinical_interpretation": {
#             "MAE_clinical": "Predictions accurate to within ~90 minutes on average",
#             "confidence": "Model explains 82% of variation in LOS",
#             "population": "10,000 Ethiopian ED patients (synthetic)"
#         },
#         "features_used": 45 if preprocessor else 0,
#         "note": "Synthetic data - validate with real Ethiopian ED data before clinical use"
#     }


# # ===== Example data for testing =====

# @app.get("/example", tags=["Examples"])
# async def get_example():
#     """Get example patient data for API testing."""
#     example_patient = {
#         "age": 35,
#         "gender": "M",
#         "chief_complaint": "Malaria",
#         "systolic_bp": 130,
#         "diastolic_bp": 85,
#         "heart_rate": 105,
#         "respiratory_rate": 22,
#         "spo2": 94,
#         "temperature_c": 39.5,
#         "triage_category": 2,
#         "region": "Addis Ababa",
#         "transport_mode": "Ambulance",
#         "imaging_ordered": "No",
#         "lab_ordered": "Basic",
#         "arrival_hour": 10
#     }
#     return {"example_patient": example_patient}


# if __name__ == "__main__":
#     import uvicorn

#     print("Starting EthioHealth-AI FastAPI Server...")
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         log_level="info"
#     )




"""
FastAPI Backend for EthioHealth-AI
Clinical Decision Support System for Ethiopian Emergency Departments
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# APP INITIALIZATION
# =========================================================
app = FastAPI(
    title="EthioHealth-AI API",
    description="AI-powered Length of Stay Prediction for Ethiopian Emergency Departments",
    version="1.0.0"
)

# =========================================================
# MODEL PATHS
# =========================================================
MODEL_DIR = Path(__file__).parent.parent / "models"

MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"

# =========================================================
# GLOBAL VARIABLES
# =========================================================
model = None
preprocessor = None

# =========================================================
# LOAD MODEL ON STARTUP
# =========================================================
@app.on_event("startup")
async def startup_event():
    global model, preprocessor

    try:

        # Load preprocessor
        if PREPROCESSOR_FILE.exists():
            preprocessor = joblib.load(PREPROCESSOR_FILE)
            print("✓ Preprocessor loaded")

        else:
            print("⚠ Preprocessor file not found")

        # Load model
        if MODEL_FILE.exists():
            model = joblib.load(MODEL_FILE)
            print("✓ XGBoost model loaded")

            if hasattr(model, "n_features_in_"):
                print(f"✓ Model expects {model.n_features_in_} features")

        else:
            print("⚠ Model file not found")

    except Exception as e:
        print(f"✗ Startup error: {e}")


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================
class PatientInput(BaseModel):
    """
    Input schema for patient LOS prediction
    """

    age: float = Field(..., gt=0, le=120)

    gender: str = Field(..., example="M")

    chief_complaint: str = Field(..., example="Malaria")

    systolic_bp: float = Field(..., ge=70, le=250)

    diastolic_bp: float = Field(..., ge=30, le=150)

    heart_rate: float = Field(..., ge=30, le=220)

    respiratory_rate: float = Field(default=20, ge=8, le=60)

    spo2: float = Field(..., ge=50, le=100)

    temperature_c: float = Field(..., ge=34, le=43)

    triage_category: int = Field(..., ge=1, le=5)

    region: str = Field(..., example="Addis Ababa")

    transport_mode: str = Field(default="Walk")

    imaging_ordered: str = Field(default="No")

    lab_ordered: str = Field(default="No")

    arrival_hour: int = Field(default=10, ge=0, le=23)


class LOSPrediction(BaseModel):

    predicted_los_hours: float

    predicted_los_category: str

    confidence_interval_lower: float

    confidence_interval_upper: float

    admission_probability: float

    risk_level: str

    recommended_action: str


class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    preprocessor_loaded: bool

    api_version: str


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def calculate_engineered_features(patient: PatientInput):

    # Peak hour logic
    is_peak_hour = int(
        (8 <= patient.arrival_hour <= 11)
        or
        (18 <= patient.arrival_hour <= 22)
    )

    # Night shift logic
    is_night_shift = int(
        patient.arrival_hour >= 22
        or
        patient.arrival_hour <= 5
    )

    # Weekend
    weekday = pd.Timestamp.now().weekday()
    is_weekend = int(weekday in [5, 6])

    # Complaint risk score
    complaint_risk_map = {
        "Trauma": 4,
        "Maternal emergency": 4,
        "Stroke": 4,
        "Sepsis": 4,
        "Malaria": 3,
        "Typhoid": 3,
        "Respiratory infection": 2,
        "Hypertensive emergency": 2,
        "Abdominal pain": 2,
        "Diarrheal disease": 1,
        "Other": 1
    }

    complaint_risk_score = complaint_risk_map.get(
        patient.chief_complaint,
        1
    )

    # Transport risk
    transport_risk_map = {
        "Walk": 1,
        "Taxi": 2,
        "Car": 2,
        "Ambulance": 4
    }

    transport_risk = transport_risk_map.get(
        patient.transport_mode,
        1
    )

    # Resource intensity
    resource_intensity = (
        int(patient.imaging_ordered != "No")
        +
        int(patient.lab_ordered != "No")
        +
        (1 if patient.triage_category <= 2 else 0)
    )

    return {
        "is_peak_hour": is_peak_hour,
        "is_night_shift": is_night_shift,
        "is_weekend": is_weekend,
        "complaint_risk_score": complaint_risk_score,
        "transport_risk": transport_risk,
        "resource_intensity": resource_intensity
    }


def preprocess_input(patient: PatientInput):

    if preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Preprocessor not loaded"
        )

    try:

        engineered = calculate_engineered_features(patient)

        data = {
            "age": [patient.age],
            "gender": [patient.gender],
            "chief_complaint": [patient.chief_complaint],
            "systolic_bp": [patient.systolic_bp],
            "diastolic_bp": [patient.diastolic_bp],
            "heart_rate": [patient.heart_rate],
            "respiratory_rate": [patient.respiratory_rate],
            "spo2": [patient.spo2],
            "temperature_c": [patient.temperature_c],
            "triage_category": [patient.triage_category],
            "region": [patient.region],
            "transport_mode": [patient.transport_mode],
            "imaging_ordered": [patient.imaging_ordered],
            "lab_ordered": [patient.lab_ordered],
            "arrival_hour": [patient.arrival_hour],

            # Engineered Features
            "is_peak_hour": [engineered["is_peak_hour"]],
            "is_night_shift": [engineered["is_night_shift"]],
            "is_weekend": [engineered["is_weekend"]],
            "complaint_risk_score": [engineered["complaint_risk_score"]],
            "transport_risk": [engineered["transport_risk"]],
            "resource_intensity": [engineered["resource_intensity"]]
        }

        df = pd.DataFrame(data)

        # Apply preprocessing
        X_processed, _ = preprocessor.prepare_dataset(
            df,
            fit=False
        )

        X = X_processed.values

        print("✓ Processed feature shape:", X.shape)

        return X

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Preprocessing failed: {str(e)}"
        )


def classify_los(hours: float):

    if hours < 4:
        return (
            "Fast-track (<4h)",
            "🟢 LOW",
            "Routine physician review"
        )

    elif hours < 12:
        return (
            "Standard (4-12h)",
            "🟡 MODERATE",
            "Observation and monitoring required"
        )

    else:
        return (
            "Extended (>12h)",
            "🔴 HIGH",
            "Urgent admission or ICU preparation"
        )


def estimate_admission_probability(
    predicted_hours: float,
    triage_category: int
):

    base_prob = {
        1: 0.95,
        2: 0.80,
        3: 0.55,
        4: 0.30,
        5: 0.15
    }.get(triage_category, 0.50)

    if predicted_hours > 12:
        prob = min(base_prob + 0.20, 1.0)

    elif predicted_hours > 8:
        prob = base_prob

    else:
        prob = max(base_prob - 0.15, 0.0)

    return round(prob, 2)


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def root():

    return {
        "message": "EthioHealth-AI API Running",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="running",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        api_version="1.0.0"
    )


@app.get("/model-info")
async def model_info():

    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": type(model).__name__ if model else None,
        "expected_features": getattr(
            model,
            "n_features_in_",
            None
        )
    }


@app.get("/example")
async def example_patient():

    return {
        "age": 35,
        "gender": "M",
        "chief_complaint": "Malaria",
        "systolic_bp": 130,
        "diastolic_bp": 85,
        "heart_rate": 105,
        "respiratory_rate": 22,
        "spo2": 94,
        "temperature_c": 39.5,
        "triage_category": 2,
        "region": "Addis Ababa",
        "transport_mode": "Ambulance",
        "imaging_ordered": "No",
        "lab_ordered": "Basic",
        "arrival_hour": 10
    }


# =========================================================
# SINGLE PREDICTION
# =========================================================
@app.post(
    "/predict",
    response_model=LOSPrediction
)
async def predict_los(patient: PatientInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:

        # Preprocess
        X = preprocess_input(patient)

        # Feature validation
        if hasattr(model, "n_features_in_"):

            expected = model.n_features_in_

            if X.shape[1] != expected:

                raise HTTPException(
                    status_code=400,
                    detail=f"""
                    Feature mismatch:
                    Model expects {expected},
                    got {X.shape[1]}
                    """
                )

        # Predict
        predicted_los = float(model.predict(X)[0])

        predicted_los = max(predicted_los, 0.5)

        # Classification
        category, risk_level, action = classify_los(
            predicted_los
        )

        # Admission probability
        admission_probability = estimate_admission_probability(
            predicted_los,
            patient.triage_category
        )

        # Confidence interval
        rmse = 1.5

        ci_lower = max(
            predicted_los - (1.96 * rmse),
            0.5
        )

        ci_upper = predicted_los + (1.96 * rmse)

        return LOSPrediction(
            predicted_los_hours=round(predicted_los, 2),
            predicted_los_category=category,
            confidence_interval_lower=round(ci_lower, 2),
            confidence_interval_upper=round(ci_upper, 2),
            admission_probability=admission_probability,
            risk_level=risk_level,
            recommended_action=action
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


# =========================================================
# BATCH PREDICTION
# =========================================================
@app.post("/batch-predict")
async def batch_predict(
    patients: List[PatientInput]
):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    results = []

    for patient in patients:

        try:

            prediction = await predict_los(patient)

            results.append({
                "patient": patient.dict(),
                "prediction": prediction.dict()
            })

        except Exception as e:

            results.append({
                "patient": patient.dict(),
                "error": str(e)
            })

    return {
        "count": len(results),
        "results": results
    }


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    import uvicorn

    print("===================================")
    print("Starting EthioHealth-AI API")
    print("===================================")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )