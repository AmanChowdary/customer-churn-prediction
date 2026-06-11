"""Tests for churn prediction model pipeline."""
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.generate_data import generate_churn_dataset
from model.train import engineer_features, preprocess

@pytest.fixture(scope="module")
def sample_df():
    return generate_churn_dataset(n=1000, seed=99)

def test_dataset_shape(sample_df):
    assert len(sample_df) == 1000
    assert "churn" in sample_df.columns

def test_churn_rate_realistic(sample_df):
    rate = sample_df["churn"].mean()
    assert 0.10 <= rate <= 0.50, f"Unexpected churn rate: {rate:.2%}"

def test_feature_engineering(sample_df):
    feat = engineer_features(sample_df)
    assert "rfm_score" in feat.columns
    assert "engagement_score" in feat.columns
    assert "contract_risk" in feat.columns
    assert "lifecycle_risk" in feat.columns
    assert "is_high_value" in feat.columns
    assert feat["rfm_score"].between(0, 1).all()

def test_preprocess_no_nulls(sample_df):
    X, y = preprocess(sample_df)
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0

def test_preprocess_output_shape(sample_df):
    X, y = preprocess(sample_df)
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    assert X.shape[1] >= 20   # at least 20 features

def test_model_training():
    """Smoke test: train on small dataset."""
    from model.train import run_training
    os.makedirs("data", exist_ok=True)
    os.makedirs("model/artifacts", exist_ok=True)
    df = generate_churn_dataset(n=2000, seed=42)
    df.to_csv("data/customer_churn.csv", index=False)
    model, metrics = run_training("data/customer_churn.csv", "model/artifacts")
    assert metrics["auc_roc"] >= 0.70, f"AUC too low: {metrics['auc_roc']}"
    assert metrics["f1"] >= 0.30

def test_prediction_output_range():
    """Model probabilities must be in [0,1]."""
    from model.train import preprocess
    import xgboost as xgb, joblib
    if not os.path.exists("model/artifacts/churn_model.joblib"):
        pytest.skip("Model not trained yet")
    model = joblib.load("model/artifacts/churn_model.joblib")
    df = generate_churn_dataset(n=100, seed=0)
    X, _ = preprocess(df)
    probs = model.predict_proba(X)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()
