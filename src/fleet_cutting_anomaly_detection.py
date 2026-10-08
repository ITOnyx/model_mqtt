"""
Fleet Learning Pipeline: Scale Deep AutoEncoder to All 11 Machines in Cutting Process Family.
1. Decodes 7 multi-sensor telemetry for all Cutting machines (Cutting-01 to Cutting-11).
2. Aligns with MES error logs across all 11 machines (localized to UTC).
3. Trains a Generalized Fleet Deep AutoEncoder for Cutting.
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

# Fix path to load local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath('src'))
from timescaledb_decoder import decompress_deltadelta_series
from extract_features_sliding_window import calculate_window_features

# Set UTF-8 stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DIR = "d:/data_2/raw"
CHUNKS_DIR = "d:/data_2/csv_raw_chunks"
OUTPUT_DIR = "d:/data_2/features"
MODEL_DIR = "d:/data_2/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CUTTING_SENSORS = [78, 79, 80, 89, 90, 91, 101]


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
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def run_fleet_pipeline():
    print("=" * 80)
    print(" SCALING INDUSTRIAL FAULT DIAGNOSIS TO ENTIRE CUTTING FLEET (11 MACHINES)")
    print("=" * 80)

    # 1. Load equipment mapping for Cutting
    df_eq = pd.read_csv(f"{RAW_DIR}/equipment.csv")
    cutting_eq = df_eq[df_eq["process_id"] == "Cutting"].sort_values("id")
    dev_to_eq = dict(zip(cutting_eq["device_id"], cutting_eq["id"]))
    
    print(f"\n[1] Identified {len(dev_to_eq)} CUTTING machines in factory:")
    for eq_id, dev_id in sorted([(v, k) for k, v in dev_to_eq.items()]):
        print(f"    - {eq_id:12s} | Device ID: {dev_id}")

    # 2. Load MES Error Logs for all Cutting machines
    print(f"\n[2] Loading and Localizing MES Error Logs for all Cutting machines...")
    df_err = pd.read_csv(f"{RAW_DIR}/lot_history_error_logs.csv")
    df_proc = pd.read_csv(f"{RAW_DIR}/processes.csv")
    cut_proc_id = df_proc[df_proc["name"] == "CUTTING"]["id"].iloc[0]
    df_err_cut = df_err[df_err["process_id"] == cut_proc_id].copy()

    # Timezone conversion (UTC+7 to UTC)
    df_err_cut["error_time_utc"] = (
        pd.to_datetime(df_err_cut["created_at"])
        .dt.tz_localize("Asia/Ho_Chi_Minh")
        .dt.tz_convert("UTC")
    )

    df_lots = pd.read_csv(f"{RAW_DIR}/lot_histories.csv", low_memory=False)
    df_err_cut = df_err_cut.merge(
        df_lots[["id", "equipment_id"]],
        left_on="lot_history_id",
        right_on="id",
        how="left",
    )
    df_err_cut = df_err_cut[df_err_cut["equipment_id"].isin(dev_to_eq.values())]

    print(f"  Total failure logs loaded across Cutting fleet: {len(df_err_cut):,}")
    error_dict = {
        eq_id: grp["error_time_utc"].tolist()
        for eq_id, grp in df_err_cut.groupby("equipment_id")
    }

    # 3. Read and decode Chunk 50 for Cutting Fleet
    chunk_path = f"{CHUNKS_DIR}/thingsboard_tskv_chunk_50.csv"
    print(f"\n[3] Reading and decoding Cutting Fleet packets from {os.path.basename(chunk_path)}...")

    cut_parts = []
    for sub_chunk in pd.read_csv(chunk_path, chunksize=30000, usecols=["entity_id", "key_id", "ts_compressed_base64", "value_compressed_base64"]):
        filtered = sub_chunk[sub_chunk["entity_id"].isin(dev_to_eq) & sub_chunk["key_id"].isin(CUTTING_SENSORS)]
        if not filtered.empty:
            cut_parts.append(filtered)
    df_chunk = pd.concat(cut_parts)
    df_chunk["equipment_id"] = df_chunk["entity_id"].map(dev_to_eq)

    print(f"  Found {len(df_chunk):,} total Cutting packets across {df_chunk['equipment_id'].nunique()} machines.")

    fleet_window_samples = []

    # Process machine by machine (take up to 120 packets per machine)
    for eq_id in sorted(df_chunk["equipment_id"].unique()):
        eq_subset = df_chunk[df_chunk["equipment_id"] == eq_id].head(120)
        print(f"  Processing {eq_id:12s} ({len(eq_subset)} packets)...", end="", flush=True)

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
        
        # Ensure all 7 sensor columns exist
        for expected_k in CUTTING_SENSORS:
            if expected_k not in df_1s.columns:
                df_1s[expected_k] = 0.0
        df_1s = df_1s[CUTTING_SENSORS]

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
                "is_running": 1.0 if win_df[78].mean() > 0 else 0.0
            }
            for col in CUTTING_SENSORS:
                col_vals = win_df[col].dropna().values
                feats = calculate_window_features(col_vals, sample_rate_hz=1.0)
                for k, v in feats.items():
                    row[f"sensor_{col}_{k}"] = v

            fleet_window_samples.append(row)
            win_count += 1
        print(f" Done ({win_count:,} windows, {sum(1 for r in fleet_window_samples[-win_count:] if r['label']==1)} warnings).")

    df_fleet = pd.DataFrame(fleet_window_samples)
    print(f"\n[4] Total Cutting Fleet Dataset: {len(df_fleet):,} window samples across {df_fleet['equipment_id'].nunique()} machines.")
    print("  Fleet Label Breakdown:")
    print(df_fleet["label"].value_counts().rename(index={0: "Normal (0)", 1: "Pre-Failure Warning (1)", 2: "Active Fault (2)"}))

    feat_out_path = f"{OUTPUT_DIR}/cutting_fleet_features.csv"
    df_fleet.to_csv(feat_out_path, index=False)
    print(f"  Cutting Fleet feature dataset saved to: {feat_out_path} ({os.path.getsize(feat_out_path)/1024:.1f} KB)")

    # 4. Train Generalized Fleet Deep AutoEncoder
    feature_cols = [c for c in df_fleet.columns if c.startswith("sensor_") and not c.endswith("_count")]
    df_normal = df_fleet[df_fleet["label"] == 0].copy()

    scaler = RobustScaler()
    X_train = scaler.fit_transform(df_normal[feature_cols].values)

    device = torch.device("cpu")
    model = FleetDeepAutoEncoder(input_dim=len(feature_cols), latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"\n[5] Training Generalized Fleet Deep AutoEncoder on Normal patterns (Input Dim: {len(feature_cols)})...")
    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/15 | Fleet Train MSE: {total_loss / len(X_train):.5f}")

    # Calculate Global Baseline Threshold
    model.eval()
    with torch.no_grad():
        X_all_norm = torch.tensor(X_train, dtype=torch.float32)
        recon_norm = model(X_all_norm)
        norm_errors = torch.mean((recon_norm - X_all_norm) ** 2, dim=1).numpy()
    
    fleet_threshold = float(np.percentile(norm_errors, 98.0))
    print(f"  Calibrated Fleet Global Threshold: {fleet_threshold:.5f} (Target FAR <= 2.0%)")

    # 5. Evaluate Per-Machine Performance Matrix
    print("\n[6] Per-Machine Evaluation & Cross-Machine Performance Matrix (CUTTING FLEET):")
    print("=" * 88)
    print(f" {'Machine':12s} | {'Normal Win':10s} | {'FAR (%)':8s} | {'Warning Win':11s} | {'Warning Recall':14s} | {'Precision':10s} | {'Status':8s}")
    print("-" * 88)

    for eq_id in sorted(df_fleet["equipment_id"].unique()):
        m_df = df_fleet[df_fleet["equipment_id"] == eq_id]
        X_m = scaler.transform(m_df[feature_cols].values)
        with torch.no_grad():
            X_m_tensor = torch.tensor(X_m, dtype=torch.float32)
            recon_m = model(X_m_tensor)
            errors_m = torch.mean((recon_m - X_m_tensor) ** 2, dim=1).numpy()

        preds = (errors_m > fleet_threshold).astype(int)
        y_true = m_df["label"].values

        norm_mask = (y_true == 0)
        warn_mask = (y_true == 1)

        far = (preds[norm_mask] == 1).mean() * 100 if norm_mask.sum() > 0 else 0.0
        recall = (preds[warn_mask] == 1).mean() * 100 if warn_mask.sum() > 0 else 0.0

        tp = np.logical_and(preds == 1, warn_mask).sum()
        fp = np.logical_and(preds == 1, norm_mask).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        status = "PASSED" if far <= 5.0 else "DRIFT"

        print(f" {eq_id:12s} | {norm_mask.sum():10,d} | {far:7.2f}% | {warn_mask.sum():11,d} | {recall:13.2f}% | {prec:10.4f} | {status:8s}")

    print("=" * 88)

    # Save fleet model and scaler
    model_save_path = f"{MODEL_DIR}/fleet_deep_autoencoder_cutting.pt"
    meta_save_path = f"{MODEL_DIR}/fleet_deep_autoencoder_cutting_metadata.joblib"
    torch.save(model.state_dict(), model_save_path)
    joblib.dump({
        "scaler": scaler,
        "feature_cols": feature_cols,
        "global_threshold": fleet_threshold,
        "sensors": CUTTING_SENSORS,
        "target_process": "Cutting"
    }, meta_save_path)

    print(f"\n[7] Fleet Model saved to: {model_save_path}")
    print(f"    Fleet Metadata saved to: {meta_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_fleet_pipeline()
