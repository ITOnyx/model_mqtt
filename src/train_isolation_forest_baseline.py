"""
Train and Evaluate Isolation Forest Baseline for Industrial Fault Early Detection.
Uses Semi-supervised paradigm with Operational Mode Awareness:
- Incorporates production state awareness (Running vs Standby).
- Trains exclusively on Normal operational data (label == 0).
- Calibrates threshold to keep False Alarm Rate (FAR) <= 2%.
- Evaluates on unseen Test Normal + Pre-Failure Warning + Active Fault periods.
- Computes Precision, Recall, F2-score, and Early Warning Lead Time.
"""

import os
import sys
import io
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, classification_report
)

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FEATURES_PATH = "d:/data_2/features/lami04_features_5sensors.csv"
MODEL_DIR = "d:/data_2/models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 75)
    print(" TRAIN & EVALUATE ISOLATION FOREST BASELINE (Lami-04 Early Anomaly Detection)")
    print("=" * 75)

    # 1. Load Data
    print("\n[1] Loading sliding window feature dataset...")
    df = pd.read_csv(FEATURES_PATH)
    df["window_end_time"] = pd.to_datetime(df["window_end_time"])
    df.sort_values("window_end_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Add Operational State Feature: is_running (1 if sensor_77_mean > 0 else 0)
    df["is_running"] = (df["sensor_77_mean"] > 0).astype(float)

    print(f"  Total samples: {len(df):,}")
    print("  Operational Breakdown:")
    print(f"    - Running (Dang san xuat): {df['is_running'].sum():,} ({df['is_running'].mean()*100:.1f}%)")
    print(f"    - Standby (Cho / Nghi ca): {(1 - df['is_running']).sum():,} ({(1 - df['is_running'].mean())*100:.1f}%)")

    # Feature columns: all sensor metrics + operational state
    feat_cols = [c for c in df.columns if c.startswith("sensor_")] + ["is_running"]

    # 2. Stratified Block Split by Operational State to prevent distribution shift
    print("\n[2] Performing State-Aware Train / Val / Test Split...")
    normal_df = df[df["label"] == 0].copy()
    anomaly_df = df[df["label"] > 0].copy()

    # Stratify normal by running state
    train_parts = []
    val_parts = []
    test_parts = []

    for state, group in normal_df.groupby("is_running"):
        n_g = len(group)
        t_end = int(0.70 * n_g)
        v_end = int(0.85 * n_g)
        train_parts.append(group.iloc[:t_end])
        val_parts.append(group.iloc[t_end:v_end])
        test_parts.append(group.iloc[v_end:])

    train_normal = pd.concat(train_parts).sort_values("window_end_time").reset_index(drop=True)
    val_normal = pd.concat(val_parts).sort_values("window_end_time").reset_index(drop=True)
    test_normal = pd.concat(test_parts).sort_values("window_end_time").reset_index(drop=True)

    # Test set combines unseen normal data + anomaly warning & fault periods
    test_df = pd.concat([test_normal, anomaly_df]).sort_values("window_end_time").reset_index(drop=True)

    print(f"  Train Set (Normal only): {len(train_normal):,} samples")
    print(f"  Validation Set (Normal only for tuning): {len(val_normal):,} samples")
    print(f"  Test Set (Real-world mix): {len(test_df):,} samples")
    print(f"    - Test Normal (label 0): {len(test_normal):,}")
    print(f"    - Test Warning (label 1): {len(df[df['label'] == 1]):,}")
    print(f"    - Test Active Fault (label 2): {len(df[df['label'] == 2]):,}")

    # 3. Scaling & Training
    print("\n[3] Fitting RobustScaler & Training Isolation Forest on Normal Baseline...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_normal[feat_cols])

    iso_forest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    print("  Model trained successfully on Normal operational patterns.")

    # 4. Threshold Calibration on Validation Set
    print("\n[4] Calibrating Anomaly Score Threshold on Validation Set...")
    X_val_scaled = scaler.transform(val_normal[feat_cols])
    val_anomaly_scores = -iso_forest.score_samples(X_val_scaled)

    # Set threshold so that False Alarm Rate (FAR) on normal validation is <= 2%
    target_far = 0.02
    calibrated_threshold = np.percentile(val_anomaly_scores, 100 * (1 - target_far))
    print(f"  Target False Alarm Rate (FAR): <= {target_far * 100:.1f}%")
    print(f"  Calibrated Anomaly Threshold: {calibrated_threshold:.4f}")

    # 5. Testing & Evaluation
    print("\n[5] Evaluating Model on Unseen Test Dataset...")
    X_test_scaled = scaler.transform(test_df[feat_cols])
    test_scores = -iso_forest.score_samples(X_test_scaled)
    test_preds = (test_scores >= calibrated_threshold).astype(int)

    # Binary evaluation: 0 = Normal, 1 = Anomaly (both Warning 1 and Fault 2)
    y_test_binary = (test_df["label"] > 0).astype(int)

    # Metrics
    precision = precision_score(y_test_binary, test_preds, zero_division=0)
    recall = recall_score(y_test_binary, test_preds, zero_division=0)
    f1 = f1_score(y_test_binary, test_preds, zero_division=0)
    f2 = fbeta_score(y_test_binary, test_preds, beta=2, zero_division=0)

    # Breakdown by sub-groups
    test_normal_mask = (test_df["label"] == 0)
    test_warning_mask = (test_df["label"] == 1)
    test_fault_mask = (test_df["label"] == 2)

    actual_far = np.mean(test_preds[test_normal_mask] == 1) * 100
    warning_recall = np.mean(test_preds[test_warning_mask] == 1) * 100
    fault_recall = np.mean(test_preds[test_fault_mask] == 1) * 100

    # Lead Time calculation for correctly predicted warnings
    correct_warning_samples = test_df[test_warning_mask & (test_preds == 1)]
    valid_lead_times = correct_warning_samples[correct_warning_samples["lead_time_seconds"] > 0]["lead_time_seconds"]
    avg_lead_time_min = (valid_lead_times.mean() / 60) if not valid_lead_times.empty else 0.0
    max_lead_time_min = (valid_lead_times.max() / 60) if not valid_lead_times.empty else 0.0

    print("\n" + "=" * 60)
    print("               MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  * False Alarm Rate (FAR) on Normal:  {actual_far:.2f}% (Chuan cong nghiep < 3%)")
    print(f"  * Early Warning Recall (Pre-Failure): {warning_recall:.2f}%")
    print(f"  * Active Fault Recall:                {fault_recall:.2f}%")
    print(f"  * Precision:                          {precision:.4f}")
    print(f"  * F1-Score:                           {f1:.4f}")
    print(f"  * F2-Score (Uu tien bat loi):         {f2:.4f}")
    print(f"  * Thoi gian canh bao truoc (Avg):    {avg_lead_time_min:.1f} phut")
    print(f"  * Thoi gian canh bao truoc (Max):    {max_lead_time_min:.1f} phut")
    print("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_test_binary, test_preds)
    print("\nConfusion Matrix [Normal vs Anomaly]:")
    print(f"  TN (Binh thuong dung):   {cm[0, 0]:,}")
    print(f"  FP (Bao dong gia):       {cm[0, 1]:,}")
    print(f"  FN (Bo sot bat thuong):  {cm[1, 0]:,}")
    print(f"  TP (Phat hien chinh xac): {cm[1, 1]:,}")

    # 6. Save Model Pipeline
    model_payload = {
        "scaler": scaler,
        "model": iso_forest,
        "calibrated_threshold": calibrated_threshold,
        "feature_names": feat_cols,
        "equipment_id": "Lami-04"
    }
    model_out_path = os.path.join(MODEL_DIR, "isolation_forest_lami04.joblib")
    joblib.dump(model_payload, model_out_path)
    print(f"\n[6] Model pipeline successfully saved to: {model_out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
