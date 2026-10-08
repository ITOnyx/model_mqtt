"""
Feature Engineering: Sliding Window Extraction on Multi-Sensor Telemetry.
Decodes all 1,397 packets for Lami-04 across 5 sensors: [77, 78, 79, 80, 101].
Extracts time-domain and frequency-domain features (RMS, Kurtosis, Crest Factor, FFT...)
and aligns with MES failure events (Normal vs Pre-failure).
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
from datetime import datetime, timezone
from timescaledb_decoder import decompress_deltadelta_series

RAW_DIR = "d:/data_2/raw"
CHUNKS_DIR = "d:/data_2/csv_raw_chunks"
OUTPUT_DIR = "d:/data_2/features"


def calculate_window_features(series: np.ndarray, sample_rate_hz: float = 1.0) -> dict:
    """
    Computes industrial time-domain and spectral features for a 1D signal window.
    """
    n = len(series)
    if n == 0 or np.all(np.isnan(series)):
        return {
            "mean": 0.0, "std": 0.0, "rms": 0.0, "p2p": 0.0,
            "crest_factor": 0.0, "kurtosis": 0.0, "skewness": 0.0,
            "fft_peak_freq": 0.0, "fft_energy": 0.0
        }
    
    # Fill remaining NaNs if any with mean
    valid_mean = np.nanmean(series)
    if np.isnan(valid_mean):
        valid_mean = 0.0
    series = np.nan_to_num(series, nan=valid_mean)

    # 1. Time-Domain Metrics
    mean_v = float(np.mean(series))
    std_v = float(np.std(series))
    rms_v = float(np.sqrt(np.mean(series ** 2)))
    p2p_v = float(np.ptp(series))
    
    peak_v = float(np.max(np.abs(series)))
    crest_factor = float(peak_v / (rms_v + 1e-8))
    
    kurt = float(stats.kurtosis(series, fisher=True)) if std_v > 1e-6 else 0.0
    skew = float(stats.skew(series)) if std_v > 1e-6 else 0.0

    # 2. Frequency-Domain Metrics (FFT)
    series_centered = series - mean_v
    fft_vals = np.abs(rfft(series_centered))
    fft_freqs = rfftfreq(n, d=1.0 / sample_rate_hz)

    fft_energy = float(np.sum(fft_vals ** 2) / (n + 1e-8))
    if len(fft_vals) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        fft_peak_freq = float(fft_freqs[peak_idx])
    else:
        fft_peak_freq = 0.0

    return {
        "mean": round(mean_v, 3),
        "std": round(std_v, 3),
        "rms": round(rms_v, 3),
        "p2p": round(p2p_v, 3),
        "crest_factor": round(crest_factor, 3),
        "kurtosis": round(kurt, 3),
        "skewness": round(skew, 3),
        "fft_peak_freq": round(fft_peak_freq, 4),
        "fft_energy": round(fft_energy, 3)
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 75)
    print(" SLIDING WINDOW FEATURE EXTRACTION - COMPLETE LAMI-04 TELEMETRY")
    print("=" * 75)

    # 1. Load Equipment Metadata
    df_eq = pd.read_csv(os.path.join(RAW_DIR, "equipment.csv"))
    device_to_eq = {}
    for _, r in df_eq[df_eq["device_id"].notna()].iterrows():
        device_to_eq[r["device_id"]] = {"equipment_id": r["id"], "name": r["name"]}

    sample_chunk_path = os.path.join(CHUNKS_DIR, "thingsboard_tskv_chunk_50.csv")
    print(f"\n[1] Reading Lami-04 packets from {os.path.basename(sample_chunk_path)} (first 2000 rows)...")
    chunk_df = pd.read_csv(sample_chunk_path, nrows=2000)
    top_entity_id = "02a134d0-da30-11f0-bdd1-bb9131434d72"
    eq_name = device_to_eq.get(top_entity_id, {}).get("name", "Unknown")
    target_eq_id = device_to_eq.get(top_entity_id, {}).get("equipment_id", "Lami-04")
    print(f"  Target: {target_eq_id} ({eq_name}) [Device ID: {top_entity_id}]")

    eq_chunks = chunk_df[chunk_df["entity_id"] == top_entity_id]
    print(f"  Total packets to decompress: {len(eq_chunks):,}")
    print(f"  Sensors present: {sorted(eq_chunks['key_id'].unique().tolist())}")

    # Decompress all records
    print("\n[2] Decompressing TimescaleDB Delta-Delta series for all sensors...")
    records = []
    for idx, (_, row) in enumerate(eq_chunks.iterrows()):
        kid = row["key_id"]
        ts_list = decompress_deltadelta_series(row["ts_compressed_base64"])
        val_list = decompress_deltadelta_series(row["value_compressed_base64"])
        for t, v in zip(ts_list, val_list):
            if t is not None and v is not None:
                records.append({"ts_ms": t, "key_id": kid, "value": v})
        if (idx + 1) % 400 == 0 or (idx + 1) == len(eq_chunks):
            print(f"    - Decompressed {idx + 1}/{len(eq_chunks)} packets ({len(records):,} data points)...")

    df_raw = pd.DataFrame(records)
    print(f"  Total raw points decoded: {len(df_raw):,}")

    df_raw["dt_utc"] = pd.to_datetime(df_raw["ts_ms"], unit="ms", utc=True)
    df_raw.sort_values("dt_utc", inplace=True)
    df_raw.drop_duplicates(subset=["key_id", "dt_utc"], inplace=True)

    # 3. Pivot to Multi-Sensor Time Series (1-second regular grid)
    print("\n[3] Resampling and Aligning multi-sensor channels onto 1-second grid...")
    df_pivot = df_raw.pivot(index="dt_utc", columns="key_id", values="value")
    # Resample to 1-second grid and forward-fill minor gaps up to 5 seconds
    df_1s = df_pivot.resample("1s").mean().ffill(limit=5)
    print(f"  Resampled grid shape: {df_1s.shape} (from {df_1s.index.min()} to {df_1s.index.max()})")

    # 4. Load Failure Events from MES for Ground Truth Alignment
    print(f"\n[4] Aligning MES Failure Events for '{target_eq_id}'...")
    df_lot = pd.read_csv(os.path.join(RAW_DIR, "lot_histories.csv"), usecols=["id", "equipment_id"])
    target_lot_ids = set(df_lot[df_lot["equipment_id"] == target_eq_id]["id"])

    df_err = pd.read_csv(os.path.join(RAW_DIR, "lot_history_error_logs.csv"))
    df_err_eq = df_err[df_err["lot_history_id"].isin(target_lot_ids)].copy()
    # MES database is recorded in Vietnam local time (UTC+7), convert to UTC
    df_err_eq["error_dt"] = pd.to_datetime(df_err_eq["created_at"]).dt.tz_localize("Asia/Ho_Chi_Minh").dt.tz_convert("UTC")
    
    error_timestamps = df_err_eq["error_dt"].dropna().sort_values().tolist()
    print(f"  Total historical error logs for {target_eq_id}: {len(error_timestamps)}")

    # 5. Sliding Window Extraction
    window_sec = 60
    step_sec = 20
    print(f"\n[5] Running Sliding Window Extraction (Window = {window_sec}s, Step = {step_sec}s)...")

    total_len = len(df_1s)
    feature_rows = []

    for start_idx in range(0, total_len - window_sec, step_sec):
        end_idx = start_idx + window_sec
        win_df = df_1s.iloc[start_idx:end_idx]
        win_end_dt = win_df.index[-1]

        # Determine Label based on Lead Time to nearest future failure:
        label = 0
        min_lead_time_sec = float("inf")
        for err_dt in error_timestamps:
            delta = (err_dt - win_end_dt).total_seconds()
            if 0 <= delta <= 1800:  # Within 30 minutes before error
                label = 1
                if delta < min_lead_time_sec:
                    min_lead_time_sec = delta
            elif -300 <= delta < 0:  # Within 5 min after error recorded
                label = 2

        row_feat = {
            "window_end_time": win_end_dt,
            "label": label,
            "lead_time_seconds": round(min_lead_time_sec, 1) if min_lead_time_sec != float("inf") else -1
        }

        # Extract features for each sensor key
        for col in df_1s.columns:
            col_data = win_df[col].dropna().values
            feats = calculate_window_features(col_data, sample_rate_hz=1.0)
            for k, val in feats.items():
                row_feat[f"sensor_{col}_{k}"] = val

        feature_rows.append(row_feat)

    df_features = pd.DataFrame(feature_rows)
    print(f"  Extracted {len(df_features):,} window samples with {df_features.shape[1]} columns each.")
    print("  Label distribution:")
    print(df_features["label"].value_counts().rename(index={0: "Normal (0)", 1: "Pre-Failure Warning (1)", 2: "Active Fault (2)"}))

    # 6. Save Extracted Features
    out_file = os.path.join(OUTPUT_DIR, "lami04_features_5sensors.csv")
    df_features.to_csv(out_file, index=False)
    print(f"\n[6] Feature dataset successfully saved to: {out_file} (Size: {os.path.getsize(out_file) / 1024:.1f} KB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
