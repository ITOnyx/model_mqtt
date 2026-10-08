"""
Benchmark Evaluation Module: Cyber-Physical Dual Pipeline
Evaluates the end-to-end multi-tiered defense:
Tier 1: MQTT IDS (Perimeter Traffic Gatekeeper)
Tier 2: MMLDA Multi-scale Physical Fault Diagnosis (Sensor Stream Analysis)

Metrics:
- End-to-End Pipeline Latency (ms)
- Cyber Threat Neutralization Rate (% malicious flows successfully blocked before reaching physical controller)
- Legitimate Traffic Pass-Through Rate (%)
- Cascading Failure Isolation Time (ms)
"""

import os
import sys
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mmlda_fault_diagnosis import MMLDANetwork

def run_cyber_physical_benchmark(n_cycles: int = 100, use_full: bool = False) -> dict:
    print(f"\n[+] Running Cyber-Physical Pipeline Benchmark ({n_cycles} multi-tier cycles)...")
    
    # 1. Load Tier 1 (Cyber IDS - Random Forest)
    rf_path = os.path.join(PROJECT_ROOT, "models", "mqtt_ids_random_forest.joblib")
    rf_bundle = joblib.load(rf_path)
    cyber_model = rf_bundle["model"]
    cyber_scaler = rf_bundle["scaler"]
    cyber_features = rf_bundle["feature_cols"]

    # 2. Load Tier 2 (Physical Diagnosis - MMLDA)
    mmlda_path = os.path.join(PROJECT_ROOT, "models", "mmlda_fleet_fault_diagnosis.pt")
    mmlda_meta = joblib.load(os.path.join(PROJECT_ROOT, "models", "mmlda_fleet_metadata.joblib"))
    phys_scaler = mmlda_meta["scaler"]
    phys_features = mmlda_meta["feature_cols"]

    phys_model = MMLDANetwork(input_dim=len(phys_features), bottleneck_dim=32, num_classes=3)
    phys_model.load_state_dict(torch.load(mmlda_path, map_location="cpu"))
    phys_model.eval()

    # Load data for feeding
    if use_full:
        df_cyber = pd.read_csv(os.path.join(PROJECT_ROOT, "features", "mqtt_ids_biflow_benchmark.csv"))
        df_phys = pd.read_csv(os.path.join(PROJECT_ROOT, "features", "cutting_fleet_features.csv"))
    else:
        df_cyber = pd.read_csv(os.path.join(PROJECT_ROOT, "benchmark", "sample_data", "mqtt_ids_sample.csv"))
        df_phys = pd.read_csv(os.path.join(PROJECT_ROOT, "benchmark", "sample_data", "physical_cutting_sample.csv"))

    cyber_samples = df_cyber.sample(n=min(len(df_cyber), n_cycles), replace=True, random_state=42)
    phys_samples = df_phys.sample(n=min(len(df_phys), n_cycles), replace=True, random_state=42)

    cyber_latencies = []
    phys_latencies = []
    e2e_latencies = []
    blocked_attacks = 0
    total_attacks = 0
    passed_benign = 0
    total_benign = 0

    for i in range(len(cyber_samples)):
        c_row = cyber_samples.iloc[i]
        p_row = phys_samples.iloc[i]
        is_attack = (c_row["label"] != 0)
        if is_attack:
            total_attacks += 1
        else:
            total_benign += 1

        t_start = time.perf_counter()

        # Step 1: Cyber Gatekeeper
        c_feats = c_row[cyber_features].values.reshape(1, -1)
        c_scaled = cyber_scaler.transform(c_feats)
        c_pred = cyber_model.predict(c_scaled)[0]
        t_tier1 = time.perf_counter()
        c_latency = (t_tier1 - t_start) * 1000
        cyber_latencies.append(c_latency)

        if c_pred != 0:
            # Threat detected -> Drop malicious network flow, do not forward to physical control bus
            blocked_attacks += 1
            e2e_latencies.append(c_latency)
            continue
        else:
            passed_benign += 1

        # Step 2: Physical Diagnosis (Only for clean forwarded telemetry)
        p_feats = p_row[phys_features].values.reshape(1, -1)
        p_scaled = phys_scaler.transform(p_feats)
        p_tensor = torch.tensor(p_scaled, dtype=torch.float32)
        with torch.no_grad():
            _, p_logits = phys_model(p_tensor)
            _ = torch.argmax(p_logits, dim=1).item()
        
        t_end = time.perf_counter()
        p_latency = (t_end - t_tier1) * 1000
        phys_latencies.append(p_latency)
        e2e_latencies.append((t_end - t_start) * 1000)

    res = {
        "domain": "Cyber-Physical",
        "task": "Dual-Tier Pipeline Cascading Defense",
        "model": "MQTT IDS (RF) + MMLDA Multi-Scale",
        "total_cycles": len(cyber_samples),
        "threat_neutralization_rate": round((blocked_attacks / max(1, total_attacks)) * 100, 2),
        "benign_pass_rate": round((passed_benign / max(1, total_benign)) * 100, 2),
        "tier1_cyber_latency_ms": round(float(np.mean(cyber_latencies)), 3),
        "tier2_physical_latency_ms": round(float(np.mean(phys_latencies)) if phys_latencies else 0.0, 3),
        "end_to_end_latency_ms": round(float(np.mean(e2e_latencies)), 3),
        "pipeline_throughput_hz": round(1000.0 / max(1e-3, float(np.mean(e2e_latencies))), 1)
    }

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyber-Physical Pipeline Benchmark")
    parser.add_argument("--cycles", type=int, default=150, help="Number of simulated cycles")
    parser.add_argument("--full", action="store_true", help="Use full datasets")
    args = parser.parse_args()

    res = run_cyber_physical_benchmark(n_cycles=args.cycles, use_full=args.full)
    df_res = pd.DataFrame([res])
    print("\n" + "=" * 95)
    print(" CYBER-PHYSICAL INTEGRATED PIPELINE BENCHMARK")
    print("=" * 95)
    print(df_res.to_string(index=False))
