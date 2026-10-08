"""
Maximum Margin Local Domain Adaptation (MMLDA) for Industrial Fault Diagnosis.
Domain Adaptation across Multi-Machine Fleet (Cutting-01 to Cutting-11).

Key Components:
1. Multi-scale Feature Extractor: Captures multi-band physical dynamics (vibration harmonics,
   temperatures, torque shifts) across 63 sensor features using multi-scale dense projections.
2. Local Domain Adaptation (LMMD - Local Maximum Mean Discrepancy):
   Aligns class-conditional distributions between Source Domain (well-annotated baseline machines)
   and Target Domain (unlabeled or differing operational machines) to eliminate domain shift.
3. Category-level Reweighting & Large-Margin Loss:
   Balances severe industrial class imbalance (Normal vs Pre-Failure vs Critical Fault)
   and forces a large decision margin to prevent negative transfer.
"""

import os
import sys
import io
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_PATH = "d:/data_2/features/cutting_fleet_features.csv"
MODEL_DIR = "d:/data_2/models"
os.makedirs(MODEL_DIR, exist_ok=True)

FAULT_CLASSES = {
    0: "Normal",
    1: "Pre-Failure Warning",
    2: "Critical Fault"
}


# ==============================================================================
# 1. Multi-scale Architecture & Maximum Margin Head
# ==============================================================================
class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale representation learner with parallel branches (32, 64, 128)
    and residual bottleneck fusion.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 32):
        super().__init__()
        self.branch_fine = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.branch_mid = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.branch_wide = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(32 + 64 + 128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_fine(x)
        b2 = self.branch_mid(x)
        b3 = self.branch_wide(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        z = self.fuse(concat)
        return z


class LargeMarginClassifier(nn.Module):
    """
    Large-Margin Classification Head.
    Forces decision boundary separation between healthy and mechanical fault states.
    """
    def __init__(self, feat_dim: int, num_classes: int = 3, margin: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, z: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.head(z)
        if labels is not None and self.training:
            one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            logits = logits - (one_hot * self.margin)
        return logits


class MMLDANetwork(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.feature_extractor = MultiScaleFeatureExtractor(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
        self.classifier = LargeMarginClassifier(feat_dim=bottleneck_dim, num_classes=num_classes, margin=0.15)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features, labels)
        return features, logits


# ==============================================================================
# 2. Local Maximum Mean Discrepancy (LMMD) Loss
# ==============================================================================
class LocalMMDLoss(nn.Module):
    """
    Class-conditional distribution alignment between source and target domain features in RKHS.
    Soft pseudo-labels from target are detached to stabilize gradient dynamics.
    """
    def __init__(self, num_classes: int = 3, kernel_mul: float = 2.0, kernel_num: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples + 1e-6)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-l2_distance / (bw + 1e-6)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(
        self,
        source_feat: torch.Tensor,
        target_feat: torch.Tensor,
        source_labels: torch.Tensor,
        target_logits: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size_s = source_feat.size(0)
        batch_size_t = target_feat.size(0)

        s_onehot = F.one_hot(source_labels, num_classes=self.num_classes).float()
        # Detach target probabilities so soft labels serve purely as conditioning weights
        t_softmax = F.softmax(target_logits.detach(), dim=-1)

        kernel_matrix = self.gaussian_kernel(source_feat, target_feat)
        k_ss = kernel_matrix[:batch_size_s, :batch_size_s]
        k_tt = kernel_matrix[batch_size_s:, batch_size_s:]
        k_st = kernel_matrix[:batch_size_s, batch_size_s:]

        loss = torch.tensor(0.0, device=source_feat.device)
        for c in range(self.num_classes):
            w_s = s_onehot[:, c].unsqueeze(1)
            w_t = t_softmax[:, c].unsqueeze(1)

            s_weight_sum = torch.sum(w_s) + 1e-6
            t_weight_sum = torch.sum(w_t) + 1e-6

            w_s_norm = w_s / s_weight_sum
            w_t_norm = w_t / t_weight_sum

            mmd_ss = torch.sum(w_s_norm @ w_s_norm.T * k_ss)
            mmd_tt = torch.sum(w_t_norm @ w_t_norm.T * k_tt)
            mmd_st = torch.sum(w_s_norm @ w_t_norm.T * k_st)

            class_disc = torch.clamp(mmd_ss + mmd_tt - 2 * mmd_st, min=0.0)
            if class_weights is not None:
                class_disc = class_disc * class_weights[c]
            loss = loss + class_disc

        return loss / self.num_classes


# ==============================================================================
# 3. Data Loading & Fleet Domain Partitioning
# ==============================================================================
def load_and_prepare_fleet_data(
    file_path: str = DATA_PATH,
    source_machines: List[str] = ["Cutting-05", "Cutting-06"],
    target_machines: List[str] = ["Cutting-02", "Cutting-03", "Cutting-04", "Cutting-10"]
) -> Tuple[Dict[str, Any], RobustScaler, List[str]]:
    print(f"[*] Reading fleet dataset from: {file_path}")
    df = pd.read_csv(file_path)

    meta_cols = ["equipment_id", "window_end_time", "label", "is_running"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"[*] Feature dimension: {len(feature_cols)} physical telemetry features.")

    # Partition into Source Domain and Target Domain
    df_src = df[df["equipment_id"].isin(source_machines)].copy()
    df_tgt = df[df["equipment_id"].isin(target_machines)].copy()

    # Fit RobustScaler strictly on Source Domain to avoid data leakage
    scaler = RobustScaler()
    x_src = scaler.fit_transform(df_src[feature_cols].values.astype(np.float32))
    y_src = df_src["label"].values.astype(np.int64)

    x_tgt = scaler.transform(df_tgt[feature_cols].values.astype(np.float32))
    y_tgt = df_tgt["label"].values.astype(np.int64)

    print(f"[*] Source domain samples ({source_machines}): {len(x_src)} | Distribution: {np.bincount(y_src)}")
    print(f"[*] Target domain samples ({target_machines}): {len(x_tgt)} | Distribution: {np.bincount(y_tgt)}")

    data_bundle = {
        "x_src": torch.tensor(x_src, dtype=torch.float32),
        "y_src": torch.tensor(y_src, dtype=torch.long),
        "x_tgt": torch.tensor(x_tgt, dtype=torch.float32),
        "y_tgt": torch.tensor(y_tgt, dtype=torch.long),
        "source_machines": source_machines,
        "target_machines": target_machines,
        "feature_cols": feature_cols
    }
    return data_bundle, scaler, feature_cols


# ==============================================================================
# 4. Training Pipeline with Category Reweighting & LMMD
# ==============================================================================
def train_mmlda_pipeline(
    data_bundle: Dict[str, Any],
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    lmmd_weight: float = 0.2,
    device: str = "cpu"
) -> Tuple[MMLDANetwork, Dict[str, Any]]:
    x_src = data_bundle["x_src"]
    y_src = data_bundle["y_src"]
    x_tgt = data_bundle["x_tgt"]
    y_tgt = data_bundle["y_tgt"]
    feature_cols = data_bundle["feature_cols"]

    # Smooth inverse-frequency class weights
    cnt = np.bincount(y_src.numpy(), minlength=3)
    smooth_weights = np.sqrt(len(y_src) / (3.0 * np.maximum(cnt, 1.0)))
    class_weights_t = torch.tensor(smooth_weights, dtype=torch.float32).to(device)
    print(f"[*] Category Reweighting factors: {dict(zip(FAULT_CLASSES.values(), np.round(smooth_weights, 3)))}")

    src_dataset = TensorDataset(x_src, y_src)
    tgt_dataset = TensorDataset(x_tgt, y_tgt)

    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim = len(feature_cols)
    model = MMLDANetwork(input_dim=input_dim, bottleneck_dim=32, num_classes=3).to(device)
    lmmd_loss_fn = LocalMMDLoss(num_classes=3)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("\n" + "=" * 80)
    print(" TRAINING MMLDA: MULTI-SCALE FEATURE LEARNING + LOCAL DOMAIN ADAPTATION")
    print("=" * 80)

    n_batches = min(len(src_loader), len(tgt_loader))
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss = 0.0
        total_lmmd_loss = 0.0
        tgt_iter = iter(tgt_loader)

        p = float(epoch) / epochs
        lambda_adapt = lmmd_weight * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

        for b_idx, (b_x_src, b_y_src) in enumerate(src_loader):
            if b_idx >= n_batches:
                break
            b_x_tgt, _ = next(tgt_iter)

            b_x_src = b_x_src.to(device)
            b_y_src = b_y_src.to(device)
            b_x_tgt = b_x_tgt.to(device)

            optimizer.zero_grad()

            feat_s, logits_s = model(b_x_src, labels=b_y_src)
            cls_loss = ce_loss_fn(logits_s, b_y_src)

            feat_t, logits_t = model(b_x_tgt)

            lmmd = lmmd_loss_fn(
                source_feat=feat_s,
                target_feat=feat_t,
                source_labels=b_y_src,
                target_logits=logits_t,
                class_weights=class_weights_t
            )

            total_loss = cls_loss + (lambda_adapt * lmmd)
            total_loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_lmmd_loss += lmmd.item()

        avg_cls = total_cls_loss / n_batches
        avg_lmmd = total_lmmd_loss / n_batches
        training_history.append({"epoch": epoch, "cls_loss": avg_cls, "lmmd_loss": avg_lmmd})

        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Cls Loss: {avg_cls:.4f} | LMMD Domain Gap: {avg_lmmd:.4f} | Lambda: {lambda_adapt:.4f}")

    # Evaluate on Target Domain
    model.eval()
    with torch.no_grad():
        _, tgt_logits = model(x_tgt.to(device))
        tgt_preds = torch.argmax(tgt_logits, dim=1).cpu().numpy()
        y_tgt_np = y_tgt.numpy()

    acc = accuracy_score(y_tgt_np, tgt_preds)
    f1_macro = f1_score(y_tgt_np, tgt_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_tgt_np, tgt_preds, average="weighted", zero_division=0)

    print("\n" + "=" * 80)
    print(" TARGET DOMAIN EVALUATION (Cross-Machine Fleet Adaptation)")
    print("=" * 80)
    print(f"Target Machines: {data_bundle['target_machines']}")
    print(f"Target Accuracy : {acc * 100:.2f}%")
    print(f"Target Macro F1 : {f1_macro * 100:.2f}%")
    print(f"Target Wgt F1   : {f1_weighted * 100:.2f}%\n")
    print("Classification Report on Target Fleet:")
    target_names = [FAULT_CLASSES[i] for i in range(3)]
    print(classification_report(y_tgt_np, tgt_preds, target_names=target_names, zero_division=0))

    metrics = {
        "target_accuracy": float(acc),
        "target_macro_f1": float(f1_macro),
        "target_weighted_f1": float(f1_weighted),
        "final_lmmd_domain_gap": float(avg_lmmd),
        "training_history": training_history
    }
    return model, metrics


# ==============================================================================
# 5. Model Persistence & Inference Utilities
# ==============================================================================
def save_mmlda_artifacts(
    model: MMLDANetwork,
    scaler: RobustScaler,
    feature_cols: List[str],
    metrics: Dict[str, Any],
    model_save_path: str = f"{MODEL_DIR}/mmlda_fleet_fault_diagnosis.pt",
    meta_save_path: str = f"{MODEL_DIR}/mmlda_fleet_metadata.joblib"
):
    print(f"[*] Saving MMLDA PyTorch weights to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

    metadata = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "fault_classes": FAULT_CLASSES,
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    print(f"[*] Saving MMLDA Metadata & Scaler to: {meta_save_path}")
    joblib.dump(metadata, meta_save_path)
    print("[+] Model artifacts saved successfully.")


class MMLDAInferenceEngine:
    """
    Lightweight inference wrapper for runtime deployment.
    Integrates directly with MQTT IDS Gateway to diagnose clean telemetry streams.
    """
    def __init__(
        self,
        model_path: str = f"{MODEL_DIR}/mmlda_fleet_fault_diagnosis.pt",
        meta_path: str = f"{MODEL_DIR}/mmlda_fleet_metadata.joblib",
        device: str = "cpu"
    ):
        self.device = device
        self.model_path = model_path
        self.meta_path = meta_path
        self.is_loaded = False
        self.load_engine()

    def load_engine(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Missing MMLDA model artifacts at {self.model_path} or {self.meta_path}")

        meta = joblib.load(self.meta_path)
        self.scaler = meta["scaler"]
        self.feature_cols = meta["feature_cols"]
        self.fault_classes = meta["fault_classes"]
        self.metrics = meta.get("metrics", {})

        self.model = MMLDANetwork(input_dim=len(self.feature_cols), bottleneck_dim=32, num_classes=3)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def diagnose_vector(self, raw_features: np.ndarray, machine_id: str = "Cutting-03") -> Dict[str, Any]:
        """
        Diagnoses a single or batch feature vector.
        Returns predicted fault class, confidence, softmax distribution, and domain shift index.
        """
        if not self.is_loaded:
            self.load_engine()

        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        scaled = self.scaler.transform(raw_features)
        tensor_x = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            feat, logits = self.model(tensor_x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            latent_norm = float(torch.norm(feat, p=2).item())

        label_name = self.fault_classes[pred_class]
        severity = "NORMAL" if pred_class == 0 else ("WARNING" if pred_class == 1 else "CRITICAL")

        return {
            "machine_id": machine_id,
            "prediction_class": pred_class,
            "prediction_label": label_name,
            "severity": severity,
            "confidence": round(confidence, 4),
            "probabilities": {self.fault_classes[i]: round(float(probs[i]), 4) for i in range(3)},
            "domain_latent_norm": round(latent_norm, 4),
            "is_anomaly": pred_class > 0,
            "recommended_action": (
                "Operational normal: No mechanical action required."
                if pred_class == 0 else
                ("Schedule inspection: Bearing vibration anomaly emerging (Pre-failure)."
                 if pred_class == 1 else
                 "EMERGENCY HALT: Critical mechanical failure imminent on spindle/bearing.")
            )
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate MMLDA Fleet Fault Diagnosis")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lmmd-weight", type=float, default=0.2, help="Local MMD loss weight")
    args = parser.parse_args()

    data_bundle, scaler, feature_cols = load_and_prepare_fleet_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training on compute device: {device}")

    model, metrics = train_mmlda_pipeline(
        data_bundle=data_bundle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lmmd_weight=args.lmmd_weight,
        device=device
    )

    save_mmlda_artifacts(model, scaler, feature_cols, metrics)

    print("\n[*] Running Sanity Test on Inference Engine...")
    engine = MMLDAInferenceEngine(device=device)
    dummy_vec = data_bundle["x_tgt"][0].numpy()
    diag = engine.diagnose_vector(dummy_vec, machine_id="Cutting-03")
    print(f"[+] Diagnostic Result: {diag}")
