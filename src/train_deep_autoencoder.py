"""
Tier 2 Model: Deep Neural AutoEncoder for Industrial Multivariate Anomaly Detection.
Trained on PyTorch (Semi-supervised on Normal baseline).
Benchmarked against Isolation Forest on:
- False Alarm Rate (FAR)
- Pre-Failure Warning Recall (Lead Time 5-30 min)
- Active Fault Recall
- F2-Score & Feature Attribution
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# Set UTF-8 stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FEATURES_PATH = "d:/data_2/features/lami04_features_5sensors.csv"
MODEL_DIR = "d:/data_2/models"


class IndustrialDeepAutoEncoder(nn.Module):
    """
    Symmetric Non-Linear AutoEncoder with Batch Normalization & GELU activations.
    Compresses multi-sensor dynamics into a compact latent bottleneck to learn
    the manifold of normal machine operation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 12):
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
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


def compute_reconstruction_error(model: nn.Module, X_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Computes Mean Squared Error (MSE) per sample across all feature dimensions."""
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_recon = model(batch_x)
            # MSE per sample
            sample_mse = torch.mean((batch_recon - batch_x) ** 2, dim=1)
            errors.extend(sample_mse.cpu().numpy())
    return np.array(errors)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 75)
    print(" TIER 2: DEEP NEURAL AUTOENCODER - INDUSTRIAL FAULT DIAGNOSIS (PyTorch)")
    print("=" * 75)

    # 1. Load Data
    print("\n[1] Loading 5-sensor feature dataset...")
    df = pd.read_csv(FEATURES_PATH)
    df["window_end_time"] = pd.to_datetime(df["window_end_time"])
    df.sort_values("window_end_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["is_running"] = (df["sensor_77_mean"] > 0).astype(float)
    feat_cols = [c for c in df.columns if c.startswith("sensor_")] + ["is_running"]
    input_dim = len(feat_cols)
    print(f"  Total samples: {len(df):,}")
    print(f"  Feature dimensions: {input_dim}")

    # 2. State-Aware Train / Val / Test Split
    print("\n[2] Splitting into Normal Baseline (Train/Val) & Test sets...")
    normal_df = df[df["label"] == 0].copy()
    anomaly_df = df[df["label"] > 0].copy()

    train_parts, val_parts, test_parts = [], [], []
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

    test_df = pd.concat([test_normal, anomaly_df]).sort_values("window_end_time").reset_index(drop=True)

    print(f"  Train Set (Normal only): {len(train_normal):,} samples")
    print(f"  Val Set (Normal only):   {len(val_normal):,} samples")
    print(f"  Test Set (Real-world):   {len(test_df):,} samples (Normal: {len(test_normal):,}, Warning: {len(df[df['label']==1]):,}, Fault: {len(df[df['label']==2]):,})")

    # 3. Preprocessing (RobustScaler)
    scaler = RobustScaler()
    X_train_np = scaler.fit_transform(train_normal[feat_cols])
    X_val_np = scaler.transform(val_normal[feat_cols])
    X_test_np = scaler.transform(test_df[feat_cols])

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

    # 4. Neural Network Training
    print("\n[3] Initializing Deep AutoEncoder Architecture & Training...")
    latent_dim = 12
    model = IndustrialDeepAutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=128, shuffle=True)
    
    epochs = 20
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x = model(batch_x)
            loss = criterion(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        
        train_loss = total_loss / len(X_train_tensor)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_tensor.to(device))
            val_loss = criterion(val_recon, X_val_tensor.to(device)).item()

        scheduler.step(val_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.5f}")

    # 5. Threshold Calibration on Normal Validation Set
    print("\n[4] Calibrating Anomaly Detection Threshold on Normal Validation Set...")
    val_errors = compute_reconstruction_error(model, X_val_tensor, device)
    
    target_far = 0.02
    threshold = float(np.percentile(val_errors, 100 * (1 - target_far)))
    print(f"  Target False Alarm Rate (FAR): <= {target_far * 100:.1f}%")
    print(f"  Calibrated Reconstruction Error Threshold: {threshold:.5f}")

    # 6. Evaluation on Unseen Test Dataset
    print("\n[5] Evaluating Deep AutoEncoder on Unseen Test Dataset...")
    test_errors = compute_reconstruction_error(model, X_test_tensor, device)
    test_preds = (test_errors >= threshold).astype(int)

    y_test_binary = (test_df["label"] > 0).astype(int)

    test_normal_mask = (test_df["label"] == 0)
    test_warning_mask = (test_df["label"] == 1)
    test_fault_mask = (test_df["label"] == 2)

    far_actual = np.mean(test_preds[test_normal_mask] == 1) * 100
    warning_recall = np.mean(test_preds[test_warning_mask] == 1) * 100
    fault_recall = np.mean(test_preds[test_fault_mask] == 1) * 100
    precision = precision_score(y_test_binary, test_preds, zero_division=0)
    f1 = f1_score(y_test_binary, test_preds, zero_division=0)
    f2 = fbeta_score(y_test_binary, test_preds, beta=2, zero_division=0)
    roc_auc = roc_auc_score(y_test_binary, test_errors)
    pr_auc = average_precision_score(y_test_binary, test_errors)

    # Lead time calculation
    correct_warning = test_df[test_warning_mask & (test_preds == 1)]
    valid_lead = correct_warning[correct_warning["lead_time_seconds"] > 0]["lead_time_seconds"]
    avg_lead_min = (valid_lead.mean() / 60) if not valid_lead.empty else 0.0
    max_lead_min = (valid_lead.max() / 60) if not valid_lead.empty else 0.0

    print("\n" + "=" * 65)
    print("          DEEP AUTOENCODER VS ISOLATION FOREST BENCHMARK")
    print("=" * 65)
    print(f"  * False Alarm Rate (FAR) on Normal:   {far_actual:6.2f}% (Chuan cong nghiep < 3%)")
    print(f"  * Early Warning Recall (Pre-Failure): {warning_recall:6.2f}%")
    print(f"  * Active Fault Recall:                 {fault_recall:6.2f}%")
    print(f"  * Precision:                           {precision:6.4f}")
    print(f"  * F1-Score:                            {f1:6.4f}")
    print(f"  * F2-Score (Uu tien bat loi):          {f2:6.4f}")
    print(f"  * ROC-AUC:                             {roc_auc:6.4f}")
    print(f"  * PR-AUC:                              {pr_auc:6.4f}")
    print(f"  * Thoi gian canh bao truoc (Avg):     {avg_lead_min:6.1f} phut")
    print(f"  * Thoi gian canh bao truoc (Max):     {max_lead_min:6.1f} phut")
    print("=" * 65)

    # Confusion matrix
    cm = confusion_matrix(y_test_binary, test_preds)
    print("\nConfusion Matrix [Normal vs Anomaly]:")
    print(f"  TN (Binh thuong dung):   {cm[0, 0]:,}")
    print(f"  FP (Bao dong gia):       {cm[0, 1]:,}")
    print(f"  FN (Bo sot bat thuong):  {cm[1, 0]:,}")
    print(f"  TP (Phat hien chinh xac): {cm[1, 1]:,}")

    # 7. Feature Reconstruction Error Attribution (Root Cause Analysis)
    print("\n[6] Feature Attribution (Top root causes of anomaly reconstruction error):")
    model.eval()
    with torch.no_grad():
        warning_x = torch.tensor(scaler.transform(df[df["label"] == 1][feat_cols]), dtype=torch.float32).to(device)
        warning_recon = model(warning_x)
        feat_mse = torch.mean((warning_recon - warning_x) ** 2, dim=0).cpu().numpy()
    
    top_indices = np.argsort(feat_mse)[::-1][:6]
    for idx in top_indices:
        print(f"  - {feat_cols[idx]:25s}: Reconstruction Error = {feat_mse[idx]:.4f}")

    # 8. Save PyTorch Artifacts
    model_save_path = os.path.join(MODEL_DIR, "deep_autoencoder_lami04.pt")
    meta_save_path = os.path.join(MODEL_DIR, "deep_autoencoder_metadata.joblib")
    torch.save(model.state_dict(), model_save_path)
    joblib.dump({
        "scaler": scaler,
        "threshold": threshold,
        "feature_names": feat_cols,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "equipment_id": "Lami-04"
    }, meta_save_path)

    print(f"\n[7] PyTorch Model weights saved to: {model_save_path}")
    print(f"    Scaler & Metadata saved to:      {meta_save_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
