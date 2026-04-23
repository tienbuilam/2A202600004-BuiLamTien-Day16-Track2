#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU ML Benchmark (LightGBM)"

# Update system and install Python packages
dnf update -y
dnf install -y python3 python3-pip git

pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy kaggle

# Create working directory
mkdir -p /home/ec2-user/ml-benchmark
chown ec2-user:ec2-user /home/ec2-user/ml-benchmark

# Write benchmark script
cat > /home/ec2-user/ml-benchmark/benchmark.py << 'PYEOF'
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print("=== LightGBM Benchmark on r5.2xlarge ===")

# Load data
t0 = time.time()
df = pd.read_csv("creditcard.csv")
load_time = time.time() - t0
print(f"Data loaded: {df.shape} rows/cols in {load_time:.2f}s")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training
params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "n_jobs": -1,
    "verbose": -1,
}

model = lgb.LGBMClassifier(**params)

t1 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)
train_time = time.time() - t1
print(f"Training time: {train_time:.2f}s | Best iteration: {model.best_iteration_}")

# Evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc    = roc_auc_score(y_test, y_pred_proba)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred)
prec   = precision_score(y_test, y_pred)
rec    = recall_score(y_test, y_pred)

print(f"AUC-ROC:   {auc:.6f}")
print(f"Accuracy:  {acc:.6f}")
print(f"F1-Score:  {f1:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall:    {rec:.6f}")

# Inference latency
single_row = X_test.iloc[:1]
times_single = []
for _ in range(100):
    t = time.time()
    model.predict_proba(single_row)
    times_single.append((time.time() - t) * 1000)
latency_1row = np.mean(times_single)

t2 = time.time()
model.predict_proba(X_test.iloc[:1000])
throughput_1000 = (time.time() - t2) * 1000

print(f"Inference latency (1 row, avg 100 runs): {latency_1row:.3f}ms")
print(f"Inference latency (1000 rows):           {throughput_1000:.3f}ms")

# Save results
results = {
    "instance_type": "r5.2xlarge",
    "dataset": "creditcard.csv",
    "n_samples": int(df.shape[0]),
    "load_time_s": round(load_time, 3),
    "train_time_s": round(train_time, 3),
    "best_iteration": int(model.best_iteration_),
    "auc_roc": round(auc, 6),
    "accuracy": round(acc, 6),
    "f1_score": round(f1, 6),
    "precision": round(prec, 6),
    "recall": round(rec, 6),
    "inference_latency_1row_ms": round(latency_1row, 3),
    "inference_latency_1000rows_ms": round(throughput_1000, 3),
}
with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== benchmark_result.json saved ===")
print(json.dumps(results, indent=2))
PYEOF

chown ec2-user:ec2-user /home/ec2-user/ml-benchmark/benchmark.py
echo "Setup complete. SSH in and run: cd ~/ml-benchmark && python3 benchmark.py"
