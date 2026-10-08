"""
Benchmark Evaluation Module: Cyber Domain (MQTT IoT IDS)
Replication of paper: "Machine Learning Based IoT Intrusion Detection System: An MQTT Case Study"

Evaluates:
1. Decision Tree (MQTT IDS)
2. Random Forest (MQTT IDS)
3. Naive Bayes Baseline (Online evaluation on benchmark split)

Metrics:
- Macro & Weighted Precision, Recall, F1-Score, Overall Accuracy
- Per-class Attack Detection Rate (Scan_A, Scan_sU, Sparta, MQTT_BF)
- Inference Latency (ms/flow) & Throughput (flows/sec)
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LABEL_MAP = {
    0: "Benign",
    1: "Scan_A",
    2: "Scan_sU",
    3: "Sparta",
    4: "MQTT_BF"
}


def evaluate_cyber_model(model_key: str, bundle: dict, df: pd.DataFrame, feature_cols: list) -> dict:
    model = bundle["model"]
    scaler = bundle["scaler"]

    X = df[feature_cols].values
    y_true = df["label"].values

    # Scaler
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    t0 = time.perf_counter()
    y_pred = model.predict(X_scaled)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class detection rate (Recall for attacks)
    per_class_rec = {}
    for c_id, c_name in LABEL_MAP.items():
        mask = (y_true == c_id)
        if mask.sum() > 0:
            per_class_rec[c_name] = round(float((y_pred[mask] == c_id).sum() / mask.sum()), 4)
        else:
            per_class_rec[c_name] = 1.0

    latency_ms = (elapsed / len(X)) * 1000
    throughput = len(X) / max(1e-6, elapsed)

    return {
        "domain": "Cyber",
        "task": "MQTT-IoT-IDS2020 Multi-Class Classification",
        "model": model_key,
        "samples": len(X),
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec_macro), 4),
        "recall_macro": round(float(rec_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "attack_dr_scan_a": per_class_rec.get("Scan_A", 0.0),
        "attack_dr_scan_su": per_class_rec.get("Scan_sU", 0.0),
        "attack_dr_sparta": per_class_rec.get("Sparta", 0.0),
        "attack_dr_mqtt_bf": per_class_rec.get("MQTT_BF", 0.0),
        "latency_ms": round(float(latency_ms), 3),
        "throughput_fps": round(float(throughput), 1)
    }


def run_cyber_benchmark(use_full: bool = False) -> list:
    print(f"\n[+] Running Cyber Domain Benchmark (Mode: {'FULL DATASET' if use_full else 'SAMPLE QUICK-TEST'})...")
    if use_full:
        data_path = os.path.join(PROJECT_ROOT, "features", "mqtt_ids_biflow_benchmark.csv")
    else:
        data_path = os.path.join(PROJECT_ROOT, "benchmark", "sample_data", "mqtt_ids_sample.csv")

    df = pd.read_csv(data_path)

    # Load trained models
    dt_path = os.path.join(PROJECT_ROOT, "models", "mqtt_ids_decision_tree.joblib")
    rf_path = os.path.join(PROJECT_ROOT, "models", "mqtt_ids_random_forest.joblib")

    dt_bundle = joblib.load(dt_path)
    rf_bundle = joblib.load(rf_path)
    feature_cols = dt_bundle["feature_cols"]

    # 1. Decision Tree
    res_dt = evaluate_cyber_model("Decision Tree", dt_bundle, df, feature_cols)

    # 2. Random Forest
    res_rf = evaluate_cyber_model("Random Forest", rf_bundle, df, feature_cols)

    # 3. Naive Bayes Baseline (Train on 50% split, test on 50% split for fair baseline)
    scaler = StandardScaler()
    X = df[feature_cols].values
    y = df["label"].values
    split_idx = int(len(df) * 0.5)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    scaler.fit(X_train)
    nb_model = GaussianNB()
    nb_model.fit(scaler.transform(X_train), y_train)

    nb_bundle = {"model": nb_model, "scaler": scaler}
    res_nb = evaluate_cyber_model("Gaussian Naive Bayes (Baseline)", nb_bundle, df.iloc[split_idx:], feature_cols)

    return [res_nb, res_dt, res_rf]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyber Domain Benchmark Runner")
    parser.add_argument("--full", action="store_true", help="Run on full dataset from features/")
    args = parser.parse_args()

    results = run_cyber_benchmark(use_full=args.full)
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print(" CYBER MQTT IDS BENCHMARK RESULTS")
    print("=" * 95)
    cols_display = ["model", "samples", "accuracy", "f1_macro", "f1_weighted", "attack_dr_mqtt_bf", "latency_ms", "throughput_fps"]
    print(df_res[cols_display].to_string(index=False))
