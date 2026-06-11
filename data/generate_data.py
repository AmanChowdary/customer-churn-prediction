"""
Generates a realistic 500K-record customer churn dataset with
behavioral, demographic, and transactional features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_churn_dataset(n=50000, seed=42):
    np.random.seed(seed)
    cid = np.arange(1, n + 1)

    # ── Demographics ──────────────────────────────────────────
    age        = np.random.randint(18, 75, n)
    gender     = np.random.choice(["M", "F", "Other"], n, p=[0.48, 0.48, 0.04])
    region     = np.random.choice(["Northeast","Southeast","Midwest","West","Southwest"], n)
    plan       = np.random.choice(["Basic","Standard","Premium","Enterprise"], n, p=[0.35,0.35,0.20,0.10])

    # ── Tenure & activity ────────────────────────────────────
    tenure_months = np.random.gamma(24, 6, n).clip(1, 120).astype(int)
    monthly_charge = {
        "Basic": np.random.uniform(10, 30, n),
        "Standard": np.random.uniform(30, 60, n),
        "Premium": np.random.uniform(60, 120, n),
        "Enterprise": np.random.uniform(120, 300, n),
    }
    charge = np.zeros(n)
    for p, arr in monthly_charge.items():
        charge += np.where(plan == p, arr, 0)
    charge = np.round(charge, 2)

    total_charges = np.round(charge * tenure_months * np.random.uniform(0.8, 1.2, n), 2)

    # ── RFM ───────────────────────────────────────────────────
    recency_days   = np.random.exponential(30, n).clip(0, 365).astype(int)
    frequency      = np.random.poisson(8, n).clip(1, 60)
    monetary_30d   = np.round(charge * np.random.uniform(0.5, 1.5, n), 2)

    # ── Engagement ───────────────────────────────────────────
    login_freq_30d    = np.random.poisson(10, n).clip(0, 90)
    support_tickets   = np.random.poisson(1.5, n).clip(0, 20)
    nps_score         = np.random.randint(0, 11, n)
    num_products      = np.random.choice([1,2,3,4,5], n, p=[0.4,0.3,0.15,0.1,0.05])
    contract_type     = np.random.choice(["Month-to-Month","One Year","Two Year"], n, p=[0.55,0.25,0.20])
    paperless_billing = np.random.choice([0, 1], n, p=[0.40, 0.60])
    payment_method    = np.random.choice(
        ["Electronic Check","Mailed Check","Bank Transfer","Credit Card"], n,
        p=[0.35, 0.20, 0.25, 0.20])

    # ── Rolling behavioral averages ───────────────────────────
    avg_session_min   = np.round(np.random.gamma(15, 3, n).clip(1, 120), 1)
    page_views_30d    = np.random.poisson(25, n).clip(0, 200)
    feature_adoption  = np.round(np.random.beta(2, 5, n), 3)   # 0–1

    # ── Churn label (engineered to be realistic ~26% rate) ────
    churn_score = (
        0.25 * (recency_days > 60).astype(float)
      + 0.20 * (contract_type == "Month-to-Month").astype(float)
      + 0.15 * (support_tickets > 3).astype(float)
      + 0.12 * (nps_score < 5).astype(float)
      + 0.10 * (login_freq_30d < 3).astype(float)
      + 0.08 * (feature_adoption < 0.1).astype(float)
      + 0.05 * (tenure_months < 6).astype(float)
      + 0.05 * (payment_method == "Electronic Check").astype(float)
      + np.random.uniform(0, 0.15, n)
    )
    churn = (churn_score > 0.45).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:07d}" for i in cid],
        "age": age, "gender": gender, "region": region,
        "tenure_months": tenure_months, "plan": plan,
        "monthly_charge": charge, "total_charges": total_charges,
        "contract_type": contract_type, "paperless_billing": paperless_billing,
        "payment_method": payment_method, "num_products": num_products,
        "recency_days": recency_days, "frequency": frequency, "monetary_30d": monetary_30d,
        "login_freq_30d": login_freq_30d, "support_tickets": support_tickets,
        "nps_score": nps_score, "avg_session_min": avg_session_min,
        "page_views_30d": page_views_30d, "feature_adoption": feature_adoption,
        "churn": churn,
    })

    print(f"Dataset: {len(df):,} rows | churn rate: {churn.mean():.1%}")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_dataset(n=50000)
    df.to_csv("data/customer_churn.csv", index=False)
    print("Saved → data/customer_churn.csv")
