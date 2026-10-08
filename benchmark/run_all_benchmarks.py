"""
Master Benchmark Orchestrator & Leaderboard Generator
Runs benchmarks across all domains:
- Physical Machine Anomaly Detection & Fleet Diagnosis
- Cyber MQTT IoT Intrusion Detection System
- Cyber-Physical Dual Pipeline

Outputs:
- Rich terminal tables
- benchmark/results/summary_leaderboard.json
- benchmark/results/summary_leaderboard.csv
- benchmark/results/plots/*.png (when --plot is enabled)
- benchmark/results/LEADERBOARD.md
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.physical.eval_physical_models import run_physical_benchmark
from benchmark.cyber.eval_mqtt_ids import run_cyber_benchmark
from benchmark.cyber_physical.eval_pipeline import run_cyber_physical_benchmark

RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def generate_plots(df_summary: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Physical Models Comparison: F1, F2 vs Latency
        df_phys = df_summary[df_summary["domain"] == "Physical"]
        if not df_phys.empty:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            x = np.arange(len(df_phys))
            width = 0.35

            rects1 = ax1.bar(x - width/2, df_phys["f1_score"], width, label="F1-Score", color="#2b5c8f")
            rects2 = ax1.bar(x + width/2, df_phys["f2_score"], width, label="F2-Score (Early Warning)", color="#e07a5f")

            ax1.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
            ax1.set_title("Physical Domain: Anomaly Detection Performance & Lead Time", fontsize=13, fontweight="bold", pad=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_phys["model"], rotation=12, ha="right", fontsize=9)
            ax1.set_ylim(0, 1.1)
            ax1.legend(loc="upper left")

            # Annotate Lead Time on bars
            for idx, r in enumerate(rects2):
                lead = df_phys["lead_time_min"].iloc[idx]
                ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03, f"{lead:.1f}m lead",
                         ha="center", va="bottom", fontsize=8, color="#333333", fontweight="bold")

            plt.tight_layout()
            phys_plot_path = os.path.join(PLOTS_DIR, "physical_models_comparison.png")
            plt.savefig(phys_plot_path, dpi=200)
            plt.close()
            print(f"[OK] Physical comparison plot saved: {phys_plot_path}")

        # 2. Cyber Models: Accuracy, F1-macro & Latency
        df_cyber = df_summary[df_summary["domain"] == "Cyber"]
        if not df_cyber.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            models = df_cyber["model"].tolist()
            accs = df_cyber["accuracy"].tolist()
            f1s = df_cyber["f1_macro"].tolist()

            y = np.arange(len(models))
            height = 0.35

            ax.barh(y - height/2, accs, height, label="Accuracy", color="#3d5a80")
            ax.barh(y + height/2, f1s, height, label="F1-Macro", color="#ee6c4d")

            ax.set_xlabel("Score", fontsize=11, fontweight="bold")
            ax.set_title("Cyber Domain: MQTT-IoT-IDS Attack Classification", fontsize=13, fontweight="bold", pad=12)
            ax.set_yticks(y)
            ax.set_yticklabels(models, fontsize=10)
            ax.set_xlim(0, 1.15)
            ax.legend(loc="lower right")

            for i in range(len(models)):
                lat = df_cyber["latency_ms"].iloc[i]
                ax.text(1.02, y[i], f"{lat:.3f} ms", va="center", fontsize=8, color="#293241", fontweight="bold")

            plt.tight_layout()
            cyber_plot_path = os.path.join(PLOTS_DIR, "cyber_ids_comparison.png")
            plt.savefig(cyber_plot_path, dpi=200)
            plt.close()
            print(f"[OK] Cyber comparison plot saved: {cyber_plot_path}")

    except Exception as e:
        print(f"[!] Warning: Plot generation failed: {e}")


def export_leaderboard_markdown(df_summary: pd.DataFrame, file_path: str):
    md = []
    md.append("# Model Benchmark Leaderboard\n")
    md.append("Bang xep hang hieu nang va thong so ky thuat cua toan bo cac mo hinh trong he thong.\n")
    md.append(f"> **Thoi diem xuat:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 1. Physical Domain
    df_phys = df_summary[df_summary["domain"] == "Physical"]
    if not df_phys.empty:
        md.append("## 1. Phan he Vat ly (Industrial Machine Fault Diagnosis)\n")
        md.append("| Mo hinh | Tac vu | Mau danh gia | Accuracy | Precision | Recall | F1-Score | F2-Score | FAR (%) | Lead Time (phut) | Latency (ms) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df_phys.iterrows():
            md.append(f"| **{r['model']}** | {r['task']} | {r['samples']} | {r.get('accuracy', '-')} | {r.get('precision', '-')} | {r.get('recall', '-')} | {r.get('f1_score', '-')} | **{r.get('f2_score', '-')}** | {r.get('far_percent', '-')} | **{r.get('lead_time_min', '-')}** | {r.get('latency_ms', '-')} |")
        md.append("\n")

    # 2. Cyber Domain
    df_cyber = df_summary[df_summary["domain"] == "Cyber"]
    if not df_cyber.empty:
        md.append("## 2. Phan he Khong gian mang (MQTT IoT IDS)\n")
        md.append("| Mo hinh | Tap kiem thu | Accuracy | F1-Macro | F1-Weighted | Attack Detection (MQTT BF) | Latency (ms) | Throughput (flows/s) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_cyber.iterrows():
            md.append(f"| **{r['model']}** | {r['samples']} | {r.get('accuracy', '-')} | **{r.get('f1_macro', '-')}** | {r.get('f1_weighted', '-')} | {r.get('attack_dr_mqtt_bf', '-')} | {r.get('latency_ms', '-')} | {r.get('throughput_fps', '-')} |")
        md.append("\n")

    # 3. Cyber-Physical
    df_cp = df_summary[df_summary["domain"] == "Cyber-Physical"]
    if not df_cp.empty:
        md.append("## 3. Phan he Tich hop (Cyber-Physical Dual-Tier Pipeline)\n")
        md.append("| Kien truc phoi hop | So chu ky test | Ty le chan tan cong (%) | Ty le thong luong hop le (%) | Tre Tier 1 Cyber (ms) | Tre Tier 2 Vat ly (ms) | Tong tre End-to-End (ms) | Tan so xu ly (Hz) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in df_cp.iterrows():
            md.append(f"| **{r['model']}** | {r['total_cycles']} | **{r.get('threat_neutralization_rate', '-')}** | {r.get('benign_pass_rate', '-')} | {r.get('tier1_cyber_latency_ms', '-')} | {r.get('tier2_physical_latency_ms', '-')} | **{r.get('end_to_end_latency_ms', '-')}** | {r.get('pipeline_throughput_hz', '-')} |")
        md.append("\n")

    md.append("---\n*Tai lieu tu dong sinh tu script `benchmark/run_all_benchmarks.py`.*")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser(description="Master Benchmark Runner for Industrial & Cyber-Physical AI")
    parser.add_argument("--domain", choices=["all", "physical", "cyber", "cyber_physical"], default="all",
                        help="Domain to benchmark (default: all)")
    parser.add_argument("--full", action="store_true", help="Run on full datasets (slower, exhaustive)")
    parser.add_argument("--plot", action="store_true", help="Generate performance visualization plots")
    parser.add_argument("--no-export", action="store_true", help="Do not export results to files")
    args = parser.parse_args()

    print("=" * 85)
    print(" INDUSTRIAL & CYBER-PHYSICAL BENCHMARK SUITE")
    print(f" Mode: {'FULL DATASET EVALUATION' if args.full else 'SAMPLE QUICK-TEST (FAST)'}")
    print(f" Target Domain: {args.domain.upper()}")
    print("=" * 85)

    all_results = []

    # 1. Physical Domain
    if args.domain in ["all", "physical"]:
        res_phys = run_physical_benchmark(use_full=args.full)
        all_results.extend(res_phys)

    # 2. Cyber Domain
    if args.domain in ["all", "cyber"]:
        res_cyber = run_cyber_benchmark(use_full=args.full)
        all_results.extend(res_cyber)

    # 3. Cyber-Physical
    if args.domain in ["all", "cyber_physical"]:
        res_cp = run_cyber_physical_benchmark(n_cycles=150, use_full=args.full)
        all_results.append(res_cp)

    df_summary = pd.DataFrame(all_results)

    # Print summary to console
    print("\n" + "=" * 85)
    print(" BENCHMARK SUMMARY LEADERBOARD")
    print("=" * 85)
    for dom in df_summary["domain"].unique():
        sub_df = df_summary[df_summary["domain"] == dom].dropna(axis=1, how="all")
        print(f"\n--- [{dom.upper()} DOMAIN] ---")
        print(sub_df.to_string(index=False))

    # Export
    if not args.no_export:
        json_path = os.path.join(RESULTS_DIR, "summary_leaderboard.json")
        csv_path = os.path.join(RESULTS_DIR, "summary_leaderboard.csv")
        md_path = os.path.join(RESULTS_DIR, "LEADERBOARD.md")

        df_summary.to_json(json_path, orient="records", indent=2)
        df_summary.to_csv(csv_path, index=False)
        export_leaderboard_markdown(df_summary, md_path)

        print("\n" + "-" * 85)
        print(f"[OK] Leaderboard JSON exported: {json_path}")
        print(f"[OK] Leaderboard CSV exported:  {csv_path}")
        print(f"[OK] Leaderboard Markdown doc:  {md_path}")

    # Plot
    if args.plot:
        print("\n[*] Generating benchmark visual comparison charts...")
        generate_plots(df_summary)

    print("\n" + "=" * 85)
    print(" BENCHMARK RUN COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
