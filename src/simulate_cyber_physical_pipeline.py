"""
End-to-End Simulation Runner for Cyber-Physical Dual Pipeline:
Demonstrates simultaneous real-time execution of:
1. Tier 1: MQTT IDS Gateway (Network Threat Mitigation & Drop)
2. Tier 2: MMLDA Fault Diagnosis (Vibration & Sensor Analysis across 11 Cutting machines)
"""

import os
import sys
import io
import time
import argparse
import asyncio
import numpy as np

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mqtt_ids_service')))

from app.model_engine import model_engine
from app.mmlda_engine import mmlda_engine
from app.clean_telemetry_forwarder import clean_forwarder
from app.security_tracker import security_tracker
from app.machine_health_tracker import machine_health_tracker
from app.flow_aggregator import FlowAggregator
from app.schemas import FlowFeatures

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


async def run_simulation_cycle(cycles: int = 6, delay: float = 0.8):
    print("=" * 85)
    print(f"{BOLD}{CYAN}CYBER-PHYSICAL DUAL PIPELINE SIMULATION: MQTT IDS + MMLDA FLEET{RESET}")
    print("=" * 85)

    print(f"[*] Initializing ML Engines...")
    model_engine.load_models()
    mmlda_engine.load_model()
    security_tracker.reset()

    print(f"[+] Tier 1 (MQTT IDS): 2 Models Loaded (Random Forest + Decision Tree)")
    print(f"[+] Tier 2 (MMLDA): Multi-Scale CNN Loaded (63 Sensor Features across 11 Machines)\n")

    scenarios = [
        {
            "name": "1. Legitimate Sensor Telemetry (Normal Operation)",
            "flow_type": "Benign",
            "source_ip": "192.168.1.101",
            "machine_id": "Cutting-01",
            "telemetry_type": "normal"
        },
        {
            "name": "2. External Cyber Attack: Aggressive TCP Port Scan (Scan_A)",
            "flow_type": "Scan_A",
            "source_ip": "10.0.0.99",
            "machine_id": "Cutting-02",
            "telemetry_type": "normal"
        },
        {
            "name": "3. Legitimate Sensor with Emerging Mechanical Bearing Anomaly",
            "flow_type": "Benign",
            "source_ip": "192.168.1.105",
            "machine_id": "Cutting-05",
            "telemetry_type": "warning"
        },
        {
            "name": "4. External Cyber Attack: MQTT Brute-Force Password Cracking (MQTT_BF)",
            "flow_type": "MQTT_BF",
            "source_ip": "10.0.0.88",
            "machine_id": "Cutting-03",
            "telemetry_type": "normal"
        },
        {
            "name": "5. Subsequent Packet from Quarantined Hacker IP",
            "flow_type": "Benign",
            "source_ip": "10.0.0.88",  # Already quarantined!
            "machine_id": "Cutting-03",
            "telemetry_type": "normal"
        },
        {
            "name": "6. Legitimate Sensor with Severe Critical Spindle Failure",
            "flow_type": "Benign",
            "source_ip": "192.168.1.108",
            "machine_id": "Cutting-08",
            "telemetry_type": "critical"
        }
    ]

    for idx, sc in enumerate(scenarios[:cycles], 1):
        print(f"\n{BOLD}{'─' * 85}{RESET}")
        print(f"{BOLD}[Event #{idx}] Scenario: {sc['name']}{RESET}")
        print(f"  Source IP: {sc['source_ip']:15s} | Target Machine: {sc['machine_id']}")

        # 1. Synthesize network flow
        flow_vec = FlowAggregator.generate_flow_by_type(sc["flow_type"])
        flow_obj = FlowFeatures(features=list(flow_vec), source_ip=sc["source_ip"])

        # 2. Synthesize physical telemetry
        telemetry = [0.0] * 63
        if sc["telemetry_type"] == "warning":
            telemetry[0] = 220.0  # RMS
            telemetry[1] = 65.0   # Std
            telemetry[2] = 8.5    # Kurtosis
        elif sc["telemetry_type"] == "critical":
            # Load real ground-truth failure sample from Cutting-02
            try:
                import pandas as pd
                df_feat = pd.read_csv("features/cutting_fleet_features.csv", nrows=16000)
                meta_cols = ["equipment_id", "window_end_time", "label", "is_running"]
                feat_cols = [c for c in df_feat.columns if c not in meta_cols]
                telemetry = df_feat.iloc[15509][feat_cols].values.tolist()
            except Exception:
                telemetry = [0.0] * 63
                telemetry[10] = 580.0

        # 3. Process via Dual Pipeline
        res = await clean_forwarder.process_and_forward(
            flow_data=flow_obj,
            telemetry_payload=telemetry,
            machine_id=sc["machine_id"]
        )

        # 4. Display Tier 1 (Cyber) Verdict
        if not res["forwarded"]:
            print(f"  {RED}[TIER 1 IDS VERDICT]{RESET} {BOLD}ATTACK DETECTED -> {res['status']}{RESET}")
            print(f"    Threat Vector   : {res.get('ids_classification', 'Unknown')}")
            print(f"    Quarantine State: {RED}Active Quarantine Enforced on {sc['source_ip']}{RESET}")
            print(f"    Action Taken    : {res.get('message')}")
            print(f"    {MAGENTA}[SAFETY SHIELD] Sensor data BLOCKED from entering MMLDA.{RESET}")
        else:
            conf_val = res['confidence'] if res['confidence'] <= 1.0 else res['confidence'] / 100.0
            print(f"  {GREEN}[TIER 1 IDS VERDICT]{RESET} Traffic Verified {BOLD}BENIGN{RESET} (Confidence: {conf_val*100:.1f}%)")
            print(f"    Broker Forward  : Forwarded to topic '{res.get('internal_topic')}'")

            # 5. Display Tier 2 (MMLDA Physical) Verdict
            diag = res["mmlda_diagnostic"]
            sev = diag["severity"]
            sev_color = GREEN if sev == "NORMAL" else (YELLOW if sev == "WARNING" else RED)
            print(f"  {sev_color}[TIER 2 MMLDA VERDICT]{RESET} Machine {BOLD}{diag['machine_id']}{RESET}: {sev_color}{diag['prediction_label']}{RESET} ({sev})")
            diag_conf = diag['confidence'] if diag['confidence'] <= 1.0 else diag['confidence'] / 100.0
            print(f"    MMLDA Confidence: {diag_conf*100:.1f}% | LMMD Latent Norm: {diag['domain_latent_norm']}")
            print(f"    Action Advisory : {diag['recommended_action']}")

        time.sleep(delay)

    # Summary
    print(f"\n{'=' * 85}")
    print(f"{BOLD}FINAL CYBER-PHYSICAL STATUS SUMMARY:{RESET}")
    print(f"{'=' * 85}")
    net_summary = security_tracker.get_status_summary()
    fleet_summary = machine_health_tracker.get_fleet_summary()

    print(f"• Total Inspected Flows: {net_summary['total_inspected']}")
    print(f"• Cyber Attacks Blocked: {RED}{net_summary['total_attacks_blocked']}{RESET} | Total Packets Dropped: {net_summary['total_packets_dropped']}")
    print(f"• Quarantined Blacklist : {[q['ip'] for q in net_summary['quarantined_ips']]}")
    print(f"• Fleet Health Score    : {GREEN}{fleet_summary['fleet_health_score']}% Healthy{RESET}")
    print(f"• Fleet Anomaly Count   : {fleet_summary['total_anomalies_detected']} physical anomalies logged across {fleet_summary['total_machines']} machines.")
    print(f"{'=' * 85}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cyber-Physical Dual Pipeline Simulation")
    parser.add_argument("--cycles", type=int, default=6, help="Number of scenarios to execute")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between events in seconds")
    args = parser.parse_args()

    asyncio.run(run_simulation_cycle(cycles=args.cycles, delay=args.delay))
