"""
Customer Churn Prediction — Training Pipeline
XGBoost + scikit-learn | SHAP explainability | MLflow tracking
"""
import os, json, warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow, mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL_COLS = ["gender", "region", "plan", "contract_type", "payment_method"]
NUMERIC_COLS = [
    "age", "tenure_months", "monthly_charge", "total_charges",
    "num_products", "paperless_billing",
    "recency_days", "frequency", "monetary_30d",
    "login_freq_30d", "support_tickets", "nps_score",
    "avg_session_min", "page_views_30d", "feature_adoption",
]
TARGET = "churn"

def engineer_features(df):
    """Construct 20+ predictive features from raw columns."""
    df = df.copy()

    # RFM composite score
    df["rfm_score"] = (
        (1 / (df["recency_days"] + 1)) * 0.4 +
        (df["frequency"] / df["frequency"].max()) * 0.3 +
        (df["monetary_30d"] / df["monetary_30d"].max()) * 0.3
    )

    # Charge-to-tenure ratio
    df["charge_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)

    # Engagement health index
    df["engagement_score"] = (
        df["login_freq_30d"] * 0.4 +
        df["page_views_30d"] * 0.3 +
        df["feature_adoption"] * 100 * 0.3
    )

    # Support burden
    df["support_per_month"] = df["support_tickets"] / df["tenure_months"].clip(lower=1)

    # Contract risk: Month-to-Month = highest churn risk
    df["contract_risk"] = df["contract_type"].map(
        {"Month-to-Month": 2, "One Year": 1, "Two Year": 0}).fillna(1)

    # NPS risk bucket
    df["nps_risk"] = pd.cut(df["nps_score"],
                             bins=[-1, 3, 6, 10], labels=[2, 1, 0]).astype(int)

    # Tenure lifecycle stage
    df["lifecycle_stage"] = pd.cut(df["tenure_months"],
        bins=[0, 3, 12, 36, 999], labels=["new","early","mature","loyal"]).astype(str)
    df["lifecycle_risk"] = df["lifecycle_stage"].map(
        {"new": 3, "early": 2, "mature": 1, "loyal": 0})

    # High-value flag
    df["is_high_value"] = (df["monthly_charge"] > df["monthly_charge"].quantile(0.75)).astype(int)

    return df

def preprocess(df):
    df = engineer_features(df)

    # Encode categoricals
    for col in CATEGORICAL_COLS + ["lifecycle_stage"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + [
        "rfm_score", "charge_per_month", "engagement_score",
        "support_per_month", "contract_risk", "nps_risk",
        "lifecycle_risk", "is_high_value", "lifecycle_stage"
    ]
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df[TARGET]
    return X, y

def train_model(X_train, y_train):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }
    print("\n── Evaluation Metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\n── Classification Report ───────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Retained","Churned"]))
    return metrics, y_prob

def plot_shap(model, X_test, output_dir):
    if not SHAP_AVAILABLE:
        print("  ⚠ SHAP not installed — skipping explainability plots.")
        return
    print("  Computing SHAP values...")
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    try:
        # Try TreeExplainer with the sklearn wrapper directly
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except (ValueError, TypeError):
        try:
            # Fallback: save/reload booster to reset internal state (fixes XGBoost 2.x / SHAP compat)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp_path = f.name
            model.save_model(tmp_path)
            import xgboost as xgb_mod
            booster = xgb_mod.Booster()
            booster.load_model(tmp_path)
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(sample)
            os.unlink(tmp_path)
        except Exception:
            # Final fallback: use XGBoost native feature importance
            print("  ⚠ SHAP TreeExplainer unavailable — using XGBoost feature_importances_ instead.")
            importances = model.feature_importances_
            top5 = sorted(zip(X_test.columns, importances), key=lambda x: x[1], reverse=True)[:5]
            print("\n── Top 5 Churn Predictors (XGBoost Gain) ────────────")
            for rank, (feat, imp) in enumerate(top5, 1):
                print(f"  {rank}. {feat}: {imp:.4f}")
            # Plot feature importance bar chart
            feat_df = pd.DataFrame({"feature": X_test.columns, "importance": importances})
            feat_df = feat_df.sort_values("importance", ascending=False).head(15)
            plt.figure(figsize=(10, 6))
            plt.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#2E75B6")
            plt.xlabel("Importance (Gain)")
            plt.title("Top 15 Churn Predictors — Feature Importance")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
            return top5

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top 5 predictors
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5 = sorted(zip(X_test.columns, mean_abs), key=lambda x: x[1], reverse=True)[:5]
    print("\n── Top 5 Churn Predictors (SHAP) ────────────────────")
    for rank, (feat, importance) in enumerate(top5, 1):
        print(f"  {rank}. {feat}: {importance:.4f}")
    return top5

def run_training(data_path="data/customer_churn.csv", model_dir="model/artifacts"):
    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} records | churn rate: {df['churn'].mean():.1%}")

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                   use_label_encoder=False, eval_metric="auc", random_state=42)
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\nTraining final model...")
    model = train_model(X_train, y_train)
    metrics, _ = evaluate(model, X_test, y_test)
    plot_shap(model, X_test, model_dir)

    # Save artifacts
    joblib.dump(model, f"{model_dir}/churn_model.joblib")
    X_train.iloc[:0].to_csv(f"{model_dir}/feature_schema.csv", index=False)
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump({**metrics, "cv_auc_mean": round(cv_scores.mean(), 4),
                   "cv_auc_std": round(cv_scores.std(), 4),
                   "trained_at": datetime.now().isoformat()}, f, indent=2)

    print(f"\n  ✓ Model saved → {model_dir}/churn_model.joblib")
    print(f"  ✓ Metrics saved → {model_dir}/metrics.json")

    # MLflow logging
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlruns/mlflow.db')}")
            mlflow.set_experiment("customer_churn_prediction")
            with mlflow.start_run():
                mlflow.log_params({"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05})
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(model, "model")
                print("  ✓ Logged to MLflow")
        except Exception as e:
            print(f"  ⚠ MLflow tracking skipped ({type(e).__name__}: {e})")

    return model, metrics

if __name__ == "__main__":
    # Auto-generate data if not present
    if not os.path.exists("data/customer_churn.csv"):
        print("Generating dataset first...")
        import sys; sys.path.insert(0, ".")
        from data.generate_data import generate_churn_dataset
        os.makedirs("data", exist_ok=True)
        generate_churn_dataset(50000).to_csv("data/customer_churn.csv", index=False)
    run_training()
