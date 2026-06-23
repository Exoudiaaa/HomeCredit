# 🏦 Home Credit Default Risk Prediction
 
End-to-end Machine Learning solution to predict the **probability of loan default** for credit applicants — from raw data ingestion to a deployed REST API microservice.
 
---
 
## 📊 Results
 
| Metric | Value |
|---|---|
| Algorithm | Logistic Regression (L2) |
| AUC-ROC | Reported via API |
| Features engineered | 264 |
| Scaled features | 63 |
| Class imbalance handling | `class_weight='balanced'` |
 
---
 
## 🔍 Project Overview
 
This project follows the **CRISP-DM** methodology across 5 phases:
 
### Phase 1 & 2 — EDA & Data Preparation
**Script:** `02_data_preparation.py`
 
- Consolidated data from `application`, `bureau` and `previous_applications`
- Feature engineering: **264 variables** including financial ratios and aggregations
- Null value treatment via **median imputation**
- Selective standardization with `StandardScaler` applied to **63 continuous variables**
### Phase 3 & 4 — Modeling & Evaluation
**Script:** `03_model_training.py`
 
- **Algorithm:** Logistic Regression with L2 penalty — chosen for its high interpretability
- Used `class_weight='balanced'` to compensate for strong class imbalance (**8% default rate**)
- **Main metric:** AUC-ROC — appropriate for imbalanced classification problems
### Phase 5 — Deployment
**Script:** `05_deployment.py`
 
- REST API built with **FastAPI**
- Internal preprocessing pipeline that:
  - Aligns any JSON input with the **264 expected features**
  - Guarantees operational robustness
- The more input fields provided, the better the prediction quality
---
 
## 🛠️ Tech Stack
 
- **Data processing:** Python, Pandas, NumPy
- **Modeling:** Scikit-learn
- **API:** FastAPI, Uvicorn
---
 
## 🚀 Getting Started
 
**1. Install dependencies**
```bash
pip install -r requirements.txt
```
 
**2. Start the API server**
```bash
python -m uvicorn 05_deployment:app --reload
```
 
**3. Open the Swagger UI**
```
http://127.0.0.1:8000/docs
```
 
---
 
## 📬 API Usage
 
Send a **POST** request to `/predecir` with a JSON body. The more fields provided, the better the prediction quality.
 
**Example request:**
```json
{
  "AMT_INCOME_TOTAL": 1200000,
  "AMT_CREDIT": 150000,
  "AMT_ANNUITY": 8000,
  "AMT_GOODS_PRICE": 150000,
  "DAYS_BIRTH": -18000,
  "DAYS_EMPLOYED": -10000,
  "EXT_SOURCE_1": 0.9,
  "EXT_SOURCE_2": 0.9,
  "EXT_SOURCE_3": 0.85,
  "REGION_POPULATION_RELATIVE": 0.04,
  "BUREAU_DAYS_CREDIT_MAX": -100,
  "BUREAU_AMT_CREDIT_SUM_MEAN": 500000,
  "BUREAU_CREDIT_COUNT": 5,
  "PREV_AMT_APPLICATION_MEAN": 100000,
  "PREV_NAME_CONTRACT_STATUS_Approved_SUM": 3,
  "PREV_COUNT": 3,
  "DAYS_REGISTRATION": -5000,
  "DAYS_ID_PUBLISH": -3000,
  "CNT_CHILDREN": 0
}
```
 
**Example response:**
```json
{
  "probabilidad_incumplimiento": 0.1453,
  "recomendacion": "APROBAR",
  "metadata": {
    "features_input": 19,
    "features_scaled": 63
  }
}
```
 
> The example above represents an **ideal applicant profile** with low default risk.
