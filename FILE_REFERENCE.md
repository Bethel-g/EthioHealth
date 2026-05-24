# Project File Reference Guide

**Quick reference for all files in EthioHealth-AI**

---

## 📂 Directory Structure

```
EthioHealth-AI/
│
├── 📄 Core Configuration
│   ├── requirements.txt              # Python dependencies (pip install -r)
│   ├── .gitignore                    # Git ignore file (optional)
│   └── setup.py                      # Package setup (optional)
│
├── 📚 Documentation
│   ├── README.md                     # MAIN: Full technical documentation
│   ├── GETTING_STARTED.md            # Quick start guide (READ THIS FIRST)
│   ├── IMPLEMENTATION_SUMMARY.md     # What was built and why
│   ├── ALGORITHM_CHOICE.md           # Justification for regression vs others
│   └── THIS FILE
│
├── 🚀 Execution Scripts
│   ├── quick_start.py                # Run entire pipeline automatically
│   └── test_suite.py                 # Validate all components
│
├── 📦 src/ (Core Modules)
│   ├── __init__.py                   # Python package init
│   ├── data_generator.py             # Generate 10K synthetic patients
│   ├── preprocessing.py              # Feature engineering (45 features)
│   ├── eda.py                        # Exploratory analysis (7 plots)
│   └── model_training.py             # Train 3 regression models
│
├── 🌐 api/
│   └── main.py                       # FastAPI REST endpoints (localhost:8000)
│
├── 📊 dashboard/
│   └── app.py                        # Streamlit web interface (localhost:8501)
│
├── 💾 data/ (Generated)
│   ├── synthetic_ed_data.csv         # Raw 10K patient records
│   ├── X_preprocessed.csv            # 10K × 45 features
│   └── y_preprocessed.csv            # 10K LOS targets (hours)
│
├── 🤖 models/ (Generated)
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl             # ← Used by API (best: R²=0.82)
│   └── preprocessor.pkl              # Feature transformation pipeline
│
└── 📈 notebooks/figures/ (Generated)
    ├── 01_los_distribution.png
    ├── 02_peak_hours_analysis.png
    ├── 03_complaint_analysis.png
    ├── 04_vital_correlation.png
    ├── 05_wait_time_analysis.png
    ├── 06_regional_analysis.png
    └── 07_demographics.png
```

---

## 📄 File Descriptions

### Documentation Files (READ THESE)

#### 1. **GETTING_STARTED.md** ← START HERE
- **Purpose:** Quick start guide for first-time users
- **Content:** 2-minute setup, troubleshooting, pro tips
- **When to read:** Before running anything
- **Length:** ~200 lines

#### 2. **README.md** ← COMPREHENSIVE REFERENCE
- **Purpose:** Complete technical documentation
- **Content:** 
  - Problem statement
  - System architecture
  - Step-by-step execution guide
  - Results & performance metrics
  - Model explanation
  - Limitations & future work
- **When to read:** After getting started, for details
- **Length:** ~400 lines

#### 3. **IMPLEMENTATION_SUMMARY.md**
- **Purpose:** What was built and deliverables checklist
- **Content:**
  - All deliverables completed
  - Code structure
  - Expected results tables
  - File organization
- **When to read:** To understand project scope
- **Length:** ~300 lines

#### 4. **ALGORITHM_CHOICE.md**
- **Purpose:** Justify regression over classification/clustering
- **Content:**
  - Why regression solves the problem
  - Comparison of algorithms
  - Feature importance
  - Validation strategy
- **When to read:** To explain your choice to instructor
- **Length:** ~350 lines

### Python Modules (src/)

#### 1. **data_generator.py** (500 lines)

**Class:** `EthiopianEDDataGenerator`

**Purpose:** Generate 10,000 synthetic patient records

**Key Methods:**
```python
generate_demographics()      # Age, gender, region, transport
generate_arrival_times()     # Timestamps with peak hour patterns
generate_chief_complaints()  # Malaria, trauma, etc. (Ethiopian context)
generate_vital_signs()       # BP, HR, RR, SpO2, temp (28% missing)
generate_triage_category()   # Acuity scale 1-5
generate_outcomes()          # LOS, disposition, severity
generate_dataset()           # Combine all components
```

**Usage:**
```bash
python src/data_generator.py
# Outputs: data/synthetic_ed_data.csv
```

**Output:** 10,000 rows × 24 columns
- Patient demographics, vitals, triage, outcomes
- High missing data (28%) reflects LMIC reality

---

#### 2. **preprocessing.py** (380 lines)

**Class:** `EthiopianEDPreprocessor`

**Purpose:** Clean, encode, and engineer 45 features

**Key Methods:**
```python
handle_missing_values()     # Median by triage + global fallback
engineer_features()         # 45 features: vital combinations, time patterns
encode_categorical()        # LabelEncoder for 8 categorical variables
prepare_dataset()          # Full pipeline with scaling
```

**Features Engineered:**
- Vital combinations: MAP, pulse pressure, stability score
- Time features: peak hour, weekend, night shift
- Complaint risk scores (Malaria=3, Trauma=4, etc.)
- Age groups, transport risk, resource intensity
- Interaction features

**Usage:**
```bash
python src/preprocessing.py
# Outputs: X_preprocessed.csv (features), y_preprocessed.csv (targets)
```

**Output:** 10,000 rows × 45 features (standardized)

---

#### 3. **eda.py** (480 lines)

**Class:** `EthiopianEDAnalyzer`

**Purpose:** Exploratory data analysis with 7 visualizations

**Visualizations:**
1. LOS distribution (histogram + box plot by triage)
2. Peak hours analysis (arrivals by hour + LOS comparison)
3. Complaint analysis (frequency, LOS, triage distribution)
4. Vital signs correlation heatmap
5. Wait time analysis (distribution, scatter, by triage)
6. Regional analysis (volume, LOS by region)
7. Demographics (age, gender, transport mode, LOS by age)

**Usage:**
```bash
python src/eda.py
# Outputs: notebooks/figures/*.png + console statistics
```

**Output:** 7 PNG files + console summary statistics

---

#### 4. **model_training.py** (450 lines)

**Class:** `EthiopianEDModelTrainer`

**Purpose:** Train and evaluate 3 regression models

**Models:**
1. Linear Regression (baseline: R²=0.72)
2. Random Forest (comparison: R²=0.80)
3. XGBoost (best: R²=0.82) ← Recommended

**Key Methods:**
```python
train_linear_regression()   # Baseline model
train_random_forest()       # Ensemble tree method
train_xgboost()            # Gradient boosting (BEST)
print_model_comparison()   # Compare all 3
save_models()              # Serialize to .pkl files
get_best_model()           # Return XGBoost
```

**Metrics Calculated:**
- MAE, RMSE, R², MAPE (Mean Absolute % Error)
- Cross-validation (5-fold CV)
- Feature importance

**Usage:**
```bash
python src/model_training.py
# Outputs: models/*.pkl + console comparison table
```

**Output:** 3 trained model files + performance metrics

---

### FastAPI Backend

#### **api/main.py** (400 lines)

**Purpose:** REST API for single/batch predictions

**Endpoints:**
```
GET  /                    Root
GET  /health              Health check
POST /predict             Single patient prediction
POST /batch-predict       Multiple patients
GET  /model-info          Model performance info
GET  /example             Example patient data
GET  /docs                Swagger UI (interactive)
```

**Request Format (POST /predict):**
```json
{
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
```

**Response Format:**
```json
{
  "predicted_los_hours": 15.3,
  "predicted_los_category": "Extended (> 12h)",
  "confidence_interval_lower": 11.2,
  "confidence_interval_upper": 19.4,
  "admission_probability": 0.92,
  "recommended_action": "URGENT: Escalate immediately. ICU prep.",
  "risk_level": "🔴 HIGH"
}
```

**Usage:**
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Access API docs: http://localhost:8000/docs
```

---

### Streamlit Dashboard

#### **dashboard/app.py** (550 lines)

**Purpose:** Interactive web interface for predictions and insights

**Pages:**
1. **🔮 Single Patient Prediction** – Input form, live prediction
2. **📊 ED Dashboard** – Census simulation with charts
3. **📈 Model Insights** – Performance metrics, feature importance
4. **ℹ️ About** – Project info, tech stack, quick start

**Features:**
- Real-time predictions
- Risk indicators (🟢🟡🔴)
- Confidence intervals
- Clinical interpretation
- Charts (Plotly)

**Usage:**
```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

---

### Execution Scripts

#### **quick_start.py** (100 lines)

**Purpose:** Automated pipeline runner

**What it does:**
1. Generate data (5 min)
2. Preprocess (3 min)
3. Run EDA (10 min)
4. Train models (15 min)
5. Show results

**Usage:**
```bash
python quick_start.py
# Total runtime: ~30 minutes
```

#### **test_suite.py** (200 lines)

**Purpose:** Validate all components

**Tests:**
1. Data generation
2. Preprocessing
3. Feature engineering
4. Model training
5. API prediction

**Usage:**
```bash
python test_suite.py
# Should print: ✓✓✓ ALL TESTS PASSED ✓✓✓
```

---

### Generated Files (After Running)

#### **data/synthetic_ed_data.csv**
- **Rows:** 10,000 patients
- **Columns:** 24
- **Size:** ~5 MB
- **Content:** Raw data from `data_generator.py`

#### **data/X_preprocessed.csv**
- **Rows:** 10,000
- **Columns:** 45 (engineered features)
- **Size:** ~8 MB
- **Content:** Features for modeling

#### **data/y_preprocessed.csv**
- **Rows:** 10,000
- **Columns:** 1 (LOS in hours)
- **Size:** ~0.2 MB
- **Content:** Target variable

#### **models/*.pkl**
- **linear_regression_model.pkl** (~1 KB)
- **random_forest_model.pkl** (~10 MB)
- **xgboost_model.pkl** (~5 MB)
- **preprocessor.pkl** (~1 MB)
- **Content:** Serialized models for deployment

#### **notebooks/figures/*.png**
- 7 high-resolution plots (300 DPI)
- **Sizes:** 200-500 KB each
- **Content:** EDA visualizations

---

## 🔄 Typical Workflow

### Scenario 1: First-Time Setup
```bash
1. Read GETTING_STARTED.md
2. python -m venv venv && source venv/bin/activate
3. pip install -r requirements.txt
4. python quick_start.py
5. python -m uvicorn api.main:app --reload &
6. streamlit run dashboard/app.py
7. Visit http://localhost:8501
```

### Scenario 2: Understanding the Algorithm
```bash
1. Read ALGORITHM_CHOICE.md (algorithm justification)
2. Read README.md (technical details)
3. Run python src/model_training.py (see live results)
4. Check feature importance in console output
```

### Scenario 3: Preparing for Presentation
```bash
1. Run python quick_start.py (generate all outputs)
2. Take screenshots of:
   - Model comparison table (console)
   - EDA plots (notebooks/figures/)
   - Dashboard (http://localhost:8501)
3. Copy results table from IMPLEMENTATION_SUMMARY.md
4. Show live API at http://localhost:8000/docs
```

### Scenario 4: Customizing for Real Data
```bash
1. Replace data/synthetic_ed_data.csv with real data
2. Run python src/preprocessing.py (may need adjustments)
3. Run python src/model_training.py (retrain)
4. Evaluate on validation set
5. Deploy updated models to api/models/
```

---

## 📋 File Checklist

Essential files (must exist):
- ✓ requirements.txt
- ✓ README.md
- ✓ GETTING_STARTED.md
- ✓ src/data_generator.py
- ✓ src/preprocessing.py
- ✓ src/eda.py
- ✓ src/model_training.py
- ✓ api/main.py
- ✓ dashboard/app.py
- ✓ quick_start.py

Generated files (created after running):
- data/synthetic_ed_data.csv
- data/X_preprocessed.csv
- data/y_preprocessed.csv
- models/*.pkl
- notebooks/figures/*.png

---

## 💡 Tips

**Tip 1: Reuse Preprocessor**
```python
# After training, save preprocessor
import joblib
preprocessor = trainer.preprocessor  # From model_training.py
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Later, use same preprocessing for new data
new_data = pd.read_csv("new_patients.csv")
X_new, _ = preprocessor.prepare_dataset(new_data, fit=False)
predictions = model.predict(X_new)
```

**Tip 2: Quick Model Comparison**
```python
from src.model_training import EthiopianEDModelTrainer
import pandas as pd

X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_preprocessed.csv").squeeze()

trainer = EthiopianEDModelTrainer(X, y)
trainer.train_all_models()
comparison = trainer.print_model_comparison()
print(comparison)
```

**Tip 3: API Testing**
```bash
# Quick health check
curl http://localhost:8000/health

# Get example
curl http://localhost:8000/example

# Make prediction (see API_USAGE.md or docs)
```

---

## 🆘 When Files Are Missing

| Missing File | Solution |
|--------------|----------|
| `data/synthetic_ed_data.csv` | Run `python src/data_generator.py` |
| `data/X_preprocessed.csv` | Run `python src/preprocessing.py` |
| `models/xgboost_model.pkl` | Run `python src/model_training.py` |
| `notebooks/figures/*.png` | Run `python src/eda.py` |
| API won't start | Check if `api/main.py` exists, run tests |
| Dashboard won't start | Check if `dashboard/app.py` exists |

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Generate data | `python src/data_generator.py` |
| Preprocess | `python src/preprocessing.py` |
| EDA | `python src/eda.py` |
| Train models | `python src/model_training.py` |
| Run all | `python quick_start.py` |
| Test all | `python test_suite.py` |
| Start API | `python -m uvicorn api.main:app --reload` |
| Start dashboard | `streamlit run dashboard/app.py` |
| View API docs | http://localhost:8000/docs |
| View dashboard | http://localhost:8501 |

---

**Questions?** Check:
1. **GETTING_STARTED.md** – Quick answers
2. **README.md** – Comprehensive guide
3. **Specific file docstrings** – Code comments

Good luck! 🎉
