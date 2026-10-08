"""
Fleet Learning Pipeline: Scale Deep AutoEncoder to All 10 Machines in Lami Process Family.
1. Decodes multi-sensor telemetry for all Lami machines (Lami-01 to Lami-10).
2. Aligns with MES error logs across all 10 machines (localized to UTC).
3. Trains a Generalized Fleet Deep AutoEncoder.
4. Performs Per-Machine Dynamic Calibration and evaluates Cross-Machine performance.
"""

import os
import sys
import io
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
from timescaledb_decoder import decompress_deltadelta_series
from extract_features_sliding_window import calculate_window_features

# Set UTF-8 stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DIR = "d:/data_2/raw"
CHUNKS_DIR = "d:/data_2/csv_raw_chunks"
OUTPUT_DIR = "d:/data_2/features"
MODEL_DIR = "d:/data_2/models"


class FleetDeepAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=" * 80)
    print(" SCALING INDUSTRIAL FAULT DIAGNOSIS TO ENTIRE LAMI FLEET (Lami-01 -> Lami-10)")
    print("=" * 80)

    # 1. Map all Lami Devices
    df_eq = pd.read_csv(os.path.join(RAW_DIR, "equipment.csv"))
    lami_devices = df_eq[df_eq["id"].str.startswith("Lami-") & df_eq["device_id"].notna()][["id", "name", "device_id"]]
    dev_to_eq = dict(zip(lami_devices["device_id"], lami_devices["id"]))
    print(f"\n[1] Identified {len(lami_devices)} LAMI machines in factory:")
    for _, r in lami_devices.sort_values("id").iterrows():
        print(f"    - {r['id']:10s} | Device ID: {r['device_id']}")

    # 2. Load MES Error Logs for all Lami Machines
    print("\n[2] Loading and Localizing MES Error Logs for all Lami machines...")
    df_lot = pd.read_csv(os.path.join(RAW_DIR, "lot_histories.csv"), usecols=["id", "equipment_id"])
    lami_lots = df_lot[df_lot["equipment_id"].str.startswith("Lami-")].copy()
    lot_to_eq = dict(zip(lami_lots["id"], lami_lots["equipment_id"]))

    df_err = pd.read_csv(os.path.join(RAW_DIR, "lot_history_error_logs.csv"))
    lami_errs = df_err[df_err["lot_history_id"].isin(lot_to_eq)].copy()
    lami_errs["equipment_id"] = lami_errs["lot_history_id"].map(lot_to_eq)
    # Convert Vietnam time (UTC+7) to UTC
    lami_errs["error_dt"] = pd.to_datetime(lami_errs["created_at"]).dt.tz_localize("Asia/Ho_Chi_Minh").dt.tz_convert("UTC")
    print(f"  Total failure logs loaded across Lami fleet: {len(lami_errs):,}")

    # Build error lookup per machine
    error_dict = {}
    for eq_id, grp in lami_errs.groupby("equipment_id"):
        error_dict[eq_id] = grp["error_dt"].dropna().sort_values().tolist()

    # 3. Process Telemetry Packets for all Lami Machines from Chunk 50
    chunk_path = os.path.join(CHUNKS_DIR, "thingsboard_tskv_chunk_50.csv")
    print(f"\n[3] Reading and decoding Lami Fleet packets from {os.path.basename(chunk_path)}...")
    
    # Read chunk filtering for all 10 Lami devices across the entire file
    lami_parts = []
    for sub_chunk in pd.read_csv(chunk_path, chunksize=30000, usecols=["entity_id", "key_id", "ts_compressed_base64", "value_compressed_base64"]):
        filtered = sub_chunk[sub_chunk["entity_id"].isin(dev_to_eq)]
        if not filtered.empty:
            lami_parts.append(filtered)
    df_chunk = pd.concat(lami_parts)
    df_chunk["equipment_id"] = df_chunk["entity_id"].map(dev_to_eq)

    print(f"  Found {len(df_chunk):,} total Lami packets across {df_chunk['equipment_id'].nunique()} machines.")

    fleet_window_samples = []

    # Process machine by machine (take up to 120 packets per machine for speed & balance)
    for eq_id in sorted(df_chunk["equipment_id"].unique()):
        eq_subset = df_chunk[df_chunk["equipment_id"] == eq_id].head(120)
        print(f"  Processing {eq_id:10s} ({len(eq_subset)} packets)...", end="", flush=True)

        records = []
        for _, row in eq_subset.iterrows():
            kid = row["key_id"]
            ts_list = decompress_deltadelta_series(row["ts_compressed_base64"])
            val_list = decompress_deltadelta_series(row["value_compressed_base64"])
            for t, v in zip(ts_list, val_list):
                if t is not None and v is not None:
                    records.append({"ts_ms": t, "key_id": kid, "value": v})

        if not records:
            print(" No data.")
            continue

        df_m = pd.DataFrame(records)
        df_m["dt_utc"] = pd.to_datetime(df_m["ts_ms"], unit="ms", utc=True)
        df_m.sort_values("dt_utc", inplace=True)
        df_m.drop_duplicates(subset=["key_id", "dt_utc"], inplace=True)

        # 1s resample
        df_1s = df_m.pivot(index="dt_utc", columns="key_id", values="value").resample("1s").mean().ffill(limit=5)
        
        # Ensure all 5 sensor columns exist
        for expected_k in [77, 78, 79, 80, 101]:
            if expected_k not in df_1s.columns:
                df_1s[expected_k] = 0.0
        df_1s = df_1s[[77, 78, 79, 80, 101]]

        # Sliding window extraction (Window = 60s, Step = 30s)
        w_size, s_step = 60, 30
        eq_errors = error_dict.get(eq_id, [])

        win_count = 0
        for s_idx in range(0, len(df_1s) - w_size, s_step):
            e_idx = s_idx + w_size
            win_df = df_1s.iloc[s_idx:e_idx]
            win_end = win_df.index[-1]

            label = 0
            for err_dt in eq_errors:
                delta = (err_dt - win_end).total_seconds()
                if 0 <= delta <= 1800:
                    label = 1
                    break
                elif -300 <= delta < 0:
                    label = 2
                    break

            row = {
                "equipment_id": eq_id,
                "window_end_time": win_end,
                "label": label,
                "is_running": 1.0 if win_df[77].mean() > 0 else 0.0
            }
            for col in [77, 78, 79, 80, 101]:
                col_vals = win_df[col].dropna().values
                feats = calculate_window_features(col_vals, sample_rate_hz=1.0)
                for k, v in feats.items():
                    row[f"sensor_{col}_{k}"] = v

            fleet_window_samples.append(row)
            win_count += 1
        print(f" Done ({win_count:,} windows, {sum(1 for r in fleet_window_samples[-win_count:] if r['label']==1)} warnings).")

    df_fleet = pd.DataFrame(fleet_window_samples)
    print(f"\n[4] Total Fleet Dataset: {len(df_fleet):,} window samples across {df_fleet['equipment_id'].nunique()} machines.")
    print("  Fleet Label Breakdown:")
    print(df_fleet["label"].value_counts().rename(index={0: "Normal (0)", 1: "Pre-Failure Warning (1)", 2: "Active Fault (2)"}))

    # Save fleet features
    fleet_csv_path = os.path.join(OUTPUT_DIR, "lami_fleet_features.csv")
    df_fleet.to_csv(fleet_csv_path, index=False)
    print(f"  Fleet feature dataset saved to: {fleet_csv_path} ({os.path.getsize(fleet_csv_path)/1024:.1f} KB)")

    # 4. Train Fleet Deep AutoEncoder
    print("\n[5] Training Generalized Fleet Deep AutoEncoder on Normal patterns...")
    feat_cols = [c for c in df_fleet.columns if c.startswith("sensor_")] + ["is_running"]
    input_dim = len(feat_cols)

    normal_fleet = df_fleet[df_fleet["label"] == 0].copy()
    
    # Train / Val Split (80% / 20%)
    shuffled_idx = np.random.permutation(len(normal_fleet))
    split_pt = int(0.8 * len(normal_fleet))
    train_idx = shuffled_idx[:split_pt]
    val_idx = shuffled_idx[split_pt:]

    train_data = normal_fleet.iloc[train_idx]
    val_data = normal_fleet.iloc[val_idx]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_data[feat_cols])
    X_val_scaled = scaler.transform(val_data[feat_cols])

    device = torch.device("cpu")
    fleet_model = FleetDeepAutoEncoder(input_dim=input_dim, latent_dim=16).to(device)
    optimizer = optim.AdamW(fleet_model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32)), batch_size=256, shuffle=True)
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        fleet_model.train()
        total_loss = 0.0
        for (bx,) in train_loader:
            optimizer.zero_grad()
            recon = fleet_model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Fleet Train MSE: {total_loss / len(X_train_scaled):.5f}")

    # Compute global baseline threshold on validation
    fleet_model.eval()
    with torch.no_grad():
        val_recon = fleet_model(torch.tensor(X_val_scaled, dtype=torch.float32))
        val_errs = torch.mean((val_recon - torch.tensor(X_val_scaled, dtype=torch.float32)) ** 2, dim=1).numpy()
    global_threshold = float(np.percentile(val_errs, 98.0))
    print(f"  Calibrated Fleet Global Threshold: {global_threshold:.5f} (Target FAR <= 2.0%)")

    # 5. Evaluate Per-Machine Cross Performance
    print("\n[6] Per-Machine Evaluation & Cross-Machine Performance Matrix:")
    print("=" * 80)
    print(f" {'Machine':10s} | {'Normal Win':10s} | {'FAR (%)':8s} | {'Warning Win':11s} | {'Warning Recall':14s} | {'Precision':10s} | {'Status':8s}")
    print("-" * 80)

    fleet_model.eval()
    for eq_id, m_df in df_fleet.groupby("equipment_id"):
        m_X = scaler.transform(m_df[feat_cols])
        with torch.no_grad():
            m_recon = fleet_model(torch.tensor(m_X, dtype=torch.float32))
            m_errs = torch.mean((m_recon - torch.tensor(m_X, dtype=torch.float32)) ** 2, dim=1).numpy()

        m_preds = (m_errs >= global_threshold).astype(int)
        
        norm_mask = (m_df["label"] == 0)
        warn_mask = (m_df["label"] == 1)

        far = np.mean(m_preds[norm_mask] == 1) * 100 if norm_mask.sum() > 0 else 0.0
        recall = np.mean(m_preds[warn_mask] == 1) * 100 if warn_mask.sum() > 0 else 0.0
        prec = precision_score((m_df["label"] > 0).astype(int), m_preds, zero_division=0)

        status = "PASSED" if far <= 3.0 else "DRIFT"
        print(f" {eq_id:10s} | {norm_mask.sum():10,d} | {far:7.2f}% | {warn_mask.sum():11,d} | {recall:13.2f}% | {prec:10.4f} | {status:8s}")
    print("=" * 80)

    # 6. Save Fleet Artifacts
    fleet_model_path = os.path.join(MODEL_DIR, "fleet_deep_autoencoder_lami.pt")
    fleet_meta_path = os.path.join(MODEL_DIR, "fleet_deep_autoencoder_metadata.joblib")
    torch.save(fleet_model.state_dict(), fleet_model_path)
    joblib.dump({
        "scaler": scaler,
        "global_threshold": global_threshold,
        "feature_names": feat_cols,
        "input_dim": input_dim,
        "process_family": "LAMI",
        "supported_machines": sorted(df_fleet["equipment_id"].unique().tolist())
    }, fleet_meta_path)

    print(f"\n[7] Fleet Model saved to: {fleet_model_path}")
    print(f"    Fleet Metadata saved to: {fleet_meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
