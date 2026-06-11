"""
Customer Churn Prediction — FastAPI REST Service
Serves real-time churn risk scores from the trained XGBoost model.
"""
import os, json
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn risk scoring powered by XGBoost + SHAP",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "model/artifacts/churn_model.joblib")
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

# ── Request / Response Schemas ────────────────────────────────

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST_0001234")
    age: int = Field(..., ge=18, le=100, example=34)
    gender: str = Field(..., example="M")
    region: str = Field(..., example="Northeast")
    tenure_months: int = Field(..., ge=0, example=24)
    plan: str = Field(..., example="Standard")
    monthly_charge: float = Field(..., ge=0, example=45.50)
    total_charges: float = Field(..., ge=0, example=1092.0)
    contract_type: str = Field(..., example="Month-to-Month")
    paperless_billing: int = Field(..., ge=0, le=1, example=1)
    payment_method: str = Field(..., example="Electronic Check")
    num_products: int = Field(..., ge=1, example=2)
    recency_days: int = Field(..., ge=0, example=45)
    frequency: int = Field(..., ge=0, example=7)
    monetary_30d: float = Field(..., ge=0, example=48.0)
    login_freq_30d: int = Field(..., ge=0, example=5)
    support_tickets: int = Field(..., ge=0, example=2)
    nps_score: int = Field(..., ge=0, le=10, example=6)
    avg_session_min: float = Field(..., ge=0, example=18.5)
    page_views_30d: int = Field(..., ge=0, example=22)
    feature_adoption: float = Field(..., ge=0.0, le=1.0, example=0.35)

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str
    top_risk_factors: List[str]
    model_version: str

class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]

# ── Feature Engineering (mirrors train.py) ───────────────────

CATEGORICAL_COLS = ["gender", "region", "plan", "contract_type", "payment_method"]
from sklearn.preprocessing import LabelEncoder

def engineer_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rfm_score"] = (
        (1 / (df["recency_days"] + 1)) * 0.4 +
        (df["frequency"] / max(df["frequency"].max(), 1)) * 0.3 +
        (df["monetary_30d"] / max(df["monetary_30d"].max(), 1)) * 0.3
    )
    df["charge_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
    df["engagement_score"] = (
        df["login_freq_30d"] * 0.4 +
        df["page_views_30d"] * 0.3 +
        df["feature_adoption"] * 100 * 0.3
    )
    df["support_per_month"] = df["support_tickets"] / df["tenure_months"].clip(lower=1)
    df["contract_risk"] = df["contract_type"].map(
        {"Month-to-Month": 2, "One Year": 1, "Two Year": 0}).fillna(1)
    df["nps_risk"] = pd.cut(df["nps_score"], bins=[-1, 3, 6, 10], labels=[2, 1, 0]).astype(int)
    df["lifecycle_stage"] = pd.cut(df["tenure_months"],
        bins=[0, 3, 12, 36, 999], labels=["new","early","mature","loyal"]).astype(str)
    df["lifecycle_risk"] = df["lifecycle_stage"].map({"new":3,"early":2,"mature":1,"loyal":0})
    df["is_high_value"] = (df["monthly_charge"] > 75).astype(int)
    for col in CATEGORICAL_COLS + ["lifecycle_stage"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def risk_tier(prob: float) -> str:
    if prob >= 0.7: return "HIGH"
    if prob >= 0.4: return "MEDIUM"
    return "LOW"

def top_risk_factors(customer: dict) -> List[str]:
    factors = []
    if customer.get("contract_type") == "Month-to-Month": factors.append("Month-to-Month contract")
    if customer.get("recency_days", 0) > 60:              factors.append("High recency (60+ days inactive)")
    if customer.get("support_tickets", 0) > 3:            factors.append("High support ticket volume")
    if customer.get("nps_score", 10) < 5:                 factors.append("Low NPS score")
    if customer.get("login_freq_30d", 10) < 3:            factors.append("Low login frequency")
    if customer.get("feature_adoption", 1) < 0.1:         factors.append("Low feature adoption")
    return factors[:3] if factors else ["No high-risk signals detected"]

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    model = load_model()
    df = pd.DataFrame([customer.dict()])
    cid = df.pop("customer_id").iloc[0]
    df = engineer_and_encode(df)

    try:
        prob = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        customer_id=cid,
        churn_probability=round(prob, 4),
        churn_prediction=int(prob >= 0.5),
        risk_tier=risk_tier(prob),
        top_risk_factors=top_risk_factors(customer.dict()),
        model_version="xgboost-v1.0",
    )

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    model = load_model()
    results = []
    df = pd.DataFrame([c.dict() for c in request.customers])
    cids = df.pop("customer_id").tolist()
    df = engineer_and_encode(df)
    probs = model.predict_proba(df)[:, 1]
    for cid, prob, cust in zip(cids, probs, request.customers):
        results.append({
            "customer_id": cid,
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(prob >= 0.5),
            "risk_tier": risk_tier(prob),
        })
    return {"predictions": results, "total": len(results)}

@app.get("/model/metrics")
async def model_metrics():
    metrics_path = "model/artifacts/metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(404, "Metrics not found. Train the model first.")
    with open(metrics_path) as f:
        return json.load(f)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
