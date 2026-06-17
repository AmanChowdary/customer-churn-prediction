"""
Render.com startup script.
Generates data + trains model if artifacts are missing, then starts the API.
Tuned for Render free tier (512 MB RAM).
"""
import os, subprocess, sys

def run(cmd):
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)

# Step 1: Generate data (10K rows — fits comfortably in 512 MB)
if not os.path.exists("data/customer_churn.csv"):
    print("Generating dataset (10K rows for free tier)...")
    os.makedirs("data", exist_ok=True)
    run("python -c \"from data.generate_data import generate_churn_dataset; generate_churn_dataset(10000).to_csv('data/customer_churn.csv', index=False)\"")

# Step 2: Train model if artifacts missing
if not os.path.exists("model/artifacts/churn_model.joblib"):
    print("Training model (~30s on free tier)...")
    run("python model/train.py")

# Step 3: Start API
port = os.getenv("PORT", "8000")
print(f"Starting API on port {port}...")
os.execvp("uvicorn", ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", port])
