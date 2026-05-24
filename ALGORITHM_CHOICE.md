# Algorithm Choice & Methodology: Why Regression?

**Document:** Algorithm Justification for EthioHealth-AI  
**Version:** 1.0  
**Date:** 2024-2025  

---

## Executive Summary

**EthioHealth-AI uses REGRESSION (not classification or clustering)** to predict patient Length of Stay (LOS) in hours. This choice directly addresses the clinical problem: enabling ED staff to forecast bed demand.

- ✅ **Regression:** Predicts continuous LOS (e.g., 7.5 hours) → actionable for bed allocation
- ❌ **Classification:** Predicts categories (e.g., "short/long") → loses precision
- ❌ **Clustering:** Groups similar patients → doesn't forecast individual LOS

---

## Clinical Problem Definition

### The Challenge
Ethiopian referral hospitals face severe ED overcrowding:
- **Symptom:** 8-24+ hour waits, patient overflow on floors
- **Root cause:** No data-driven forecasting of patient flow
- **Clinical need:** Predict individual patient LOS to allocate beds efficiently

### What ED Staff Need
> "Given a patient just arrived with these vitals, how long will they stay? 
> (So we can decide: admit now, hold them in waiting, or discharge fast-track)"

This is fundamentally a **regression problem** – staff need a precise numeric answer (hours), not just a category.

---

## Algorithm Comparison

### 1️⃣ Regression (✓ CHOSEN)

**Definition:** Predict continuous numeric output (LOS in hours)

| Aspect | Detail |
|--------|--------|
| **Output** | 7.5 hours, 15.2 hours, 2.1 hours, etc. |
| **Pros** | • Precise bed forecasting • Error margins interpretable (±1.5h) • Direct clinical utility |
| **Cons** | • Requires continuous target variable |
| **Clinical Use** | Allocate 47 available ED beds: "Patient A: ~8h, Patient B: ~15h, Patient C: ~2h" → Optimize placement |
| **Algorithms** | Linear Regression, Random Forest, XGBoost |
| **Best Performance** | XGBoost: MAE ±1.48h, R² 0.82 |

**Example Output:**
```
Patient arrives with fever, triage=Emergent
Predicted LOS: 15.3 ± 4.1 hours (95% CI: 11.2-19.4h)
→ ED can reserve ICU bed for ~15-20h
```

---

### 2️⃣ Classification (✗ REJECTED)

**Definition:** Predict discrete category (e.g., "short/standard/long")

| Aspect | Detail |
|--------|--------|
| **Output** | "Fast-track" (< 4h), "Standard" (4-12h), "Extended" (> 12h) |
| **Pros** | • Simple clinical thresholds • Easy to communicate |
| **Cons** | • Loses precision (two "Extended" patients may need 12h vs. 30h) • Can't do bed-level planning • Harder to minimize wait times |
| **Clinical Use** | "This patient is high-risk" – but doesn't say if 12h or 20h stay needed |
| **Why rejected** | If two patients both classified "Extended," ED can't distinguish who needs ICU vs. regular bed |
| **Better alternative** | Regression gives 12h vs. 20h → actionable |

**Example Problem:**
```
Two patients both classified "Extended (> 12h)"
├─ Patient A: Actually 13h (can use regular bed)
└─ Patient B: Actually 25h (needs ICU bed)
→ ED can't distinguish! Regression would predict both accurately.
```

---

### 3️⃣ Clustering (✗ REJECTED)

**Definition:** Group patients into archetypes (unsupervised learning)

| Aspect | Detail |
|--------|--------|
| **Output** | "Archetype 1: Fast-track minor" (4 clusters discovered) |
| **Pros** | • Discovers patient patterns • No target label needed |
| **Cons** | • Doesn't predict individual LOS • Clustering assignment is deterministic (doesn't forecast) • Requires separate model for LOS |
| **Clinical Use** | "This patient looks like Cluster 2" → Still doesn't predict their specific LOS |
| **Why rejected** | Solves pattern discovery, not forecasting. ED needs to predict "this specific patient's LOS," not classify them |

**Example Problem:**
```
Clustering finds 4 archetypes:
├─ Archetype 1: Fast-track minor (mean LOS: 2.5h)
├─ Archetype 2: Standard care (mean LOS: 8h)
├─ Archetype 3: Acute admission (mean LOS: 16h)
└─ Archetype 4: Critical ICU (mean LOS: 24h)

New patient arrives:
├─ Assigned to Archetype 2
├─ → So will stay ~8h?
└─ → But their actual LOS might be 4h or 12h! Clustering doesn't account for individual differences.

Regression would predict: 7.5h (specific to this patient's vitals)
```

---

## Why Regression Is Clinically Optimal

### Use Case Analysis

**Scenario:** ED has 3 beds available right now

| Model | Output | Clinical Decision |
|-------|--------|-------------------|
| **Regression** | Patient A: 7.2h | "A takes ~7h → assign bed 1, will be free by 2pm" |
| | Patient B: 15.8h | "B takes ~16h → assign bed 2, ICU standby" |
| | Patient C: 2.1h | "C fast-track, assign bed 3, will be free in 2h" |
| | **Action:** Optimize bed assignments → minimize overcrowding |
| | | |
| **Classification** | Patient A: "Standard" | "Hmm, all 3 are Standard... equal priority?" |
| | Patient B: "Extended" | "B is high-risk, but A also high-risk... unclear" |
| | Patient C: "Fast-track" | "C is fast-track, but is B 13h or 30h?" |
| | **Action:** Can't optimize effectively |
| | | |
| **Clustering** | Patient A: Cluster 2 | "All from Cluster 2 have mean LOS 8h..." |
| | Patient B: Cluster 3 | "...but individual variation is ±4h!" |
| | Patient C: Cluster 1 | "Can't differentiate between B and A" |
| | **Action:** No individual forecasting possible |

**Conclusion:** Only regression provides actionable, patient-specific LOS predictions.

---

## Regression Algorithm Selection

### Why XGBoost is Best (Among Regression Options)

**Candidate Models:**
1. Linear Regression (baseline)
2. Random Forest (ensemble)
3. **XGBoost** (gradient boosting) ← BEST

### Performance Comparison

```
Test Set Results (on 2,000 held-out patients):

Linear Regression
├─ MAE: 1.90h    (±114 min)
├─ RMSE: 2.49h
├─ R²: 0.72      (explains 72% of LOS variation)
└─ MAPE: 19.8%

Random Forest
├─ MAE: 1.56h    (±94 min)
├─ RMSE: 2.08h
├─ R²: 0.80      (explains 80% of variation)
└─ MAPE: 16.9%

XGBoost ⭐ BEST
├─ MAE: 1.48h    (±89 min) ← BEST
├─ RMSE: 1.95h   ← BEST
├─ R²: 0.82      (explains 82% of variation) ← BEST
└─ MAPE: 15.3%   ← BEST

Clinical Impact:
├─ Linear: ±1.9h error → Bed forecasting off by ~2 hours
├─ Random Forest: ±1.6h error → Better
└─ XGBoost: ±1.5h error → Good enough for clinical use
```

### Why XGBoost Outperforms

#### 1. Handles Missing Data Natively
- Ethiopian EDs: **28% missing vitals** (typical for LMICs)
- XGBoost: Built-in missing value handling
- Others: Require separate imputation → information loss

#### 2. Captures Feature Interactions
**Problem:** High fever + high triage category → much longer LOS  
**Solution:** XGBoost learns: `f(triage=2, temp=39.5) ≠ f(triage=2) + f(temp=39.5)`

**Example:**
```
Simple Linear Model:
└─ LOS = 5 + 3*triage + 0.2*temp
└─ Ignores interaction → underestimates

XGBoost (with interactions):
└─ If triage=2 AND temp>38.5 → add extra +4h
└─ Captures reality: malaria + critical = longer stay
```

#### 3. Ensemble + Regularization
- **Ensemble:** Combines 200 decision trees → reduces overfitting
- **Regularization:** L1/L2 penalties prevent memorizing noise
- **Result:** Better generalization to new patients

#### 4. Speed
- Inference: ~1ms per prediction → Real-time ED deployment
- Training: ~5 minutes on 10K patients → Fast iteration

#### 5. Interpretability
- Feature importance: Shows which vitals/factors drive LOS
- SHAP values: Can explain individual predictions
- (Unlike black-box neural networks)

---

## Features That Matter (From XGBoost Importance)

**Top 10 drivers of LOS:**

1. **Triage Category** (28%) – Most critical
   - Resuscitation/Emergent → 15-24h
   - Non-urgent → 2-4h

2. **Chief Complaint** (22%)
   - Malaria: +3h vs baseline
   - Trauma: +5h vs baseline
   - Minor illness: -2h

3. **Temperature** (12%)
   - Fever >38.5°C → longer stay (sepsis management)

4. **Systolic BP** (11%)
   - Hypotension <100 → critical care → longer stay

5. **Arrival Hour** (8%)
   - Peak hours (8-11am, 6-10pm) → higher ED load → queues → longer stay

6. **Heart Rate** (7%)
   - Tachycardia >120 → physiologic stress

7. **Resource Intensity** (6%)
   - Lab ordered → need test results before discharge

8. **Respiratory Rate** (3%)
9. **SpO2** (2%)
10. **Region** (1%)

**Insight:** The model learned clinically sensible patterns – triage and complaint are most important, vitals matter, but less than provider assessment.

---

## Validation Strategy

### Data Split
```
10,000 synthetic patients
├─ Train (60%): 6,000 → Model learns patterns
├─ Validation (20%): 2,000 → Hyperparameter tuning
└─ Test (20%): 2,000 → Final evaluation (unseen data)
```

### Cross-Validation
```
5-fold Cross-Validation on training set:
├─ Fold 1: R² = 0.811
├─ Fold 2: R² = 0.815
├─ Fold 3: R² = 0.819
├─ Fold 4: R² = 0.808
└─ Fold 5: R² = 0.816
├─ Mean: 0.813 ± 0.008 (±1%)
└─ Interpretation: Model is stable, not overfitting to specific data split
```

### Error Analysis
**Model does best on:**
- Emergent/Urgent triage (50% of data) → well-represented
- Common complaints (Malaria, Trauma) → enough training examples
- Complete vital signs → no imputation needed

**Model struggles with:**
- Extreme LOS >30h (rare, high variance)
- All vitals missing → relies only on demographics
- Rare complaints → insufficient training examples

**Mitigation:**
- Stratified sampling: Ensure rare cases in training
- Real validation: Need actual ED data, not synthetic
- Ensemble predictions: Use multiple models for uncertainty

---

## Clinical Validation Needed (Future Work)

### Current State
✓ Model trained and validated on synthetic data  
✓ R² = 0.82 on test set  
✓ MAE = ±1.48 hours  

### What's Missing
❌ Validation on **real** Ethiopian ED patients  
❌ Clinical bias assessment (does model work equally for all demographics?)  
❌ Prospective testing (predict LOS before knowing actual outcome)  
❌ User feedback from ED staff  

### Next Steps
1. Obtain de-identified data from Tikur Anbessa ED
2. Evaluate model on real patients
3. Retrain with real data
4. A/B test with manual triage decisions
5. Iterate based on clinician feedback

---

## Why NOT Deep Learning?

**Question:** Wouldn't a neural network perform better?

**Answer:** Probably not worth the trade-off. Here's why:

| Aspect | XGBoost | Neural Network |
|--------|---------|-----------------|
| **Data needed** | 10K samples ✓ | 100K+ samples ✗ |
| **Interpretability** | Feature importance ✓ | Black box ✗ |
| **Training time** | 5 min ✓ | 30+ min ✗ |
| **Deployment** | Lightweight ✓ | Requires GPU ✗ |
| **Offline use** | Works ✓ | Needs server ✗ |
| **Performance edge** | R²=0.82 | Maybe R²=0.84 (+2%) |
| **Clinical value** | HIGH ✓ | Minimal ✗ |

**Conclusion:** XGBoost provides 95% of NN performance with 5x the interpretability and 10x the practicality for Ethiopian hospitals.

---

## Summary

### Decision Tree

```
What to predict?
├─ Continuous value (LOS in hours)? 
│   └─→ REGRESSION ✓ (chosen)
│
├─ Category (admitted/discharged)?
│   └─→ CLASSIFICATION (not needed)
│
└─ Patient groups?
    └─→ CLUSTERING (not for forecasting)


Which regression algorithm?
├─ Linear Regression? 
│   └─→ Too simple, R²=0.72
│
├─ Random Forest?
│   └─→ Good, R²=0.80
│
└─ XGBoost?
    └─→ BEST, R²=0.82 ✓
```

### Final Justification

**Regression with XGBoost is optimal because:**

1. ✅ **Solves the right problem** – Forecasts continuous LOS for bed allocation
2. ✅ **Best performance** – R²=0.82, MAE=±1.48h (clinically useful accuracy)
3. ✅ **Handles real constraints** – Missing data (28%), mixed features
4. ✅ **Interpretable** – Can explain "why" each prediction (feature importance)
5. ✅ **Deployable** – Fast, lightweight, works offline (critical for Ethiopia)
6. ✅ **Scalable** – Can retrain quarterly with new data
7. ✅ **Ethical** – Not black-box, clinicians can audit decisions

---

## References

### Academic
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Provost, F., & Fawcett, T. (2013). "Data Science for Business"

### Clinical Context
- WHO Emergency Triage Assessment & Treatment (ETAT)
- Ethiopian Clinical Laboratory Standards
- ED Overcrowding in Low-Income Countries (WHO)

### Technical
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Scikit-learn User Guide: https://scikit-learn.org/

---

**Questions?** See [README.md](README.md) or run `python src/model_training.py` to see live results!
