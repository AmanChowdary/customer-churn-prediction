# Customer Churn Prediction & Analysis

Supervised ML pipeline predicting customer churn with 91% AUC-ROC. Deployed as a FastAPI REST service with SHAP explainability and real-time risk scoring.

## Architecture

```
[Raw Data] ──► [Feature Engineering (20+ features)] ──► [XGBoost Classifier]
                                                                  │
                                                           [SHAP Explainer]
                                                                  │
                                                       [FastAPI REST Service]
                                                                  │
                                                       [Power BI Dashboard]
```

## Key Features
- **91% AUC-ROC** on 500K+ records using XGBoost with engineered behavioral features
- **20+ predictive features**: RFM scores, rolling averages, lifecycle stage indicators
- **SHAP explainability**: top 5 behavioral churn predictors surfaced for product team
- **FastAPI REST API**: real-time single + batch prediction endpoints
- **Docker containerized** for production deployment

## Quick Start

```bash
pip install -r requirements.txt

# Generate data & train model
python data/generate_data.py
python model/train.py

# Start API
uvicorn api.main:app --reload --port 8000
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"CUST_001","age":34,"gender":"M","region":"Northeast",
       "tenure_months":6,"plan":"Basic","monthly_charge":25.0,"total_charges":150.0,
       "contract_type":"Month-to-Month","paperless_billing":1,
       "payment_method":"Electronic Check","num_products":1,
       "recency_days":75,"frequency":3,"monetary_30d":25.0,
       "login_freq_30d":2,"support_tickets":4,"nps_score":3,
       "avg_session_min":8.0,"page_views_30d":5,"feature_adoption":0.05}'

# Model metrics
curl http://localhost:8000/model/metrics
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Run Tests
```bash
pytest tests/ -v --cov=model
```

## Tech Stack
`Python` `XGBoost` `scikit-learn` `SHAP` `FastAPI` `Docker` `MLflow` `pandas` `GitHub Actions`
