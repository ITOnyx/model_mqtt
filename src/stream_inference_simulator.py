"""
Industrial Real-time Stream Inference Simulator & Alert Dashboard.
Simulates incoming 1-second telemetry stream from ThingsBoard,
computes sliding-window features, infers reconstruction error via Deep AutoEncoder,
detects Pre-Failure anomalies, identifies Root Cause sensors, and renders a live CLI dashboard.
"""

import os
import sys
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fleet_cutting_anomaly_detection import FleetDeepAutoEncoder

# ANSI terminal colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_bar(val: float, max_val: float, width: int = 25) -> str:
    filled = int(min(1.0, val / max_val) * width) if max_val > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return bar


def main():
    parser = argparse.ArgumentParser(description="Real-Time Industrial Stream Inference Simulator")
    parser.add_argument("--machine", type=str, default="Cutting-03", help="Target equipment ID")
    parser.add_argument("--speed", type=float, default=0.35, help="Simulation delay per window in seconds")
    parser.add_argument("--limit", type=int, default=30, help="Number of streaming cycles to simulate")
    args = parser.parse_args()

    # Load Model & Metadata
    model_path = "models/fleet_deep_autoencoder_cutting.pt"
    meta_path = "models/fleet_deep_autoencoder_cutting_metadata.joblib"

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print(f"{RED}[ERROR] Model or metadata file not found in models/{RESET}")
        return

    meta = joblib.load(meta_path)
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]
    threshold = meta["global_threshold"]

    model = FleetDeepAutoEncoder(input_dim=len(feature_cols), latent_dim=16)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load Feature Stream for Target Machine
    feat_file = "features/cutting_fleet_features.csv"
    if not os.path.exists(feat_file):
        print(f"{RED}[ERROR] Feature file {feat_file} not found.{RESET}")
        return

    df = pd.read_csv(feat_file)
    m_df = df[df["equipment_id"] == args.machine].copy().reset_index(drop=True)

    if m_df.empty:
        print(f"{RED}[ERROR] No data found for machine {args.machine}{RESET}")
        return

    # Find transition point from Normal to Pre-Failure Warning to demonstrate early detection
    warn_indices = m_df[m_df["label"] == 1].index
    if len(warn_indices) > 0:
        start_idx = max(0, warn_indices[0] - 5)
    else:
        start_idx = 0

    stream_slice = m_df.iloc[start_idx: start_idx + args.limit].copy().reset_index(drop=True)

    print(f"\n{BOLD}{CYAN}========================================================================================{RESET}")
    print(f"{BOLD}{CYAN}      INDUSTRIAL REAL-TIME STREAM INFERENCE SIMULATOR & ALERT DASHBOARD (AI MES)       {RESET}")
    print(f"{BOLD}{CYAN}========================================================================================{RESET}")
    print(f" Target Equipment : {BOLD}{args.machine}{RESET} | Process Family: CUTTING")
    print(f" Active Sensors   : 7 Channels [78, 79, 80, 89, 90, 91, 101] (63 Extracted Features)")
    print(f" Anomaly Model    : Deep AutoEncoder (16-Dim Latent Bottleneck, GELU, BatchNorm)")
    print(f" Safety Threshold : {BOLD}{threshold:.2f}{RESET} MSE")
    print(f"{CYAN}----------------------------------------------------------------------------------------{RESET}")
    print(f" Streaming simulation starting in 2 seconds... (Press Ctrl+C to abort)\n")
    time.sleep(2)

    # Streaming Loop
    for cycle, (_, row) in enumerate(stream_slice.iterrows(), start=1):
        x_raw = row[feature_cols].values.reshape(1, -1)
        x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = scaler.transform(x_raw)

        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32)
            recon_t = model(x_t)
            feat_errors = (recon_t - x_t).squeeze(0).numpy() ** 2
            recon_error = float(np.mean(feat_errors))

        # Root Cause Analysis: Top 2 contributing sensors
        sensor_error_map = {}
        for idx, col in enumerate(feature_cols):
            # Parse sensor ID
            parts = col.split("_")
            s_name = f"Sensor_{parts[1]}"
            sensor_error_map[s_name] = sensor_error_map.get(s_name, 0.0) + feat_errors[idx]

        sorted_sensors = sorted(sensor_error_map.items(), key=lambda item: item[1], reverse=True)
        top1_sensor, top1_err = sorted_sensors[0]
        top2_sensor, top2_err = sorted_sensors[1]

        # Determine Alert Status & Lead Time
        ground_truth = int(row["label"])
        is_alert = recon_error > threshold
        ratio = recon_error / threshold

        if ratio >= 1.5:
            health_color = RED
            status_text = "CRITICAL ANOMALY DETECTED"
            action_text = f"HALT / INSPECT {top1_sensor.upper()}"
        elif ratio >= 1.0:
            health_color = YELLOW
            status_text = "PRE-FAILURE WARNING (EARLY ALERT)"
            action_text = f"SCHEDULE MAINTENANCE (EST LEAD: 15-30m)"
        else:
            health_color = GREEN
            status_text = "SYSTEM NOMINAL (NORMAL STATE)"
            action_text = "NO ACTION REQUIRED"

        bar = format_bar(recon_error, threshold * 2.0, width=20)
        timestamp_str = str(row["window_end_time"])[:19]

        print(f"[{BOLD}{cycle:02d}/{args.limit}{RESET}] Time: {timestamp_str} | Err: {recon_error:8.1f} [{bar}] {ratio:4.2f}x Thresh")
        print(f"     Status    : {health_color}{BOLD}{status_text}{RESET}")
        print(f"     MES Truth : {'Normal (0)' if ground_truth==0 else ('Pre-Failure (1)' if ground_truth==1 else 'Active Fault (2)')}")
        print(f"     Root Cause: {top1_sensor} (MSE: {top1_err:7.1f}) | {top2_sensor} (MSE: {top2_err:7.1f})")
        print(f"     Action    : {BOLD}{action_text}{RESET}")
        print(f"     " + "-" * 80)

        time.sleep(args.speed)

    print(f"\n{BOLD}{GREEN}✔ Stream simulation finished successfully across {args.limit} telemetry windows.{RESET}\n")


if __name__ == "__main__":
    main()
