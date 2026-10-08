# Model Benchmark Leaderboard

Bang xep hang hieu nang va thong so ky thuat cua toan bo cac mo hinh trong he thong.

> **Thoi diem xuat:** 2026-10-08 09:12:36


## 1. Phan he Vat ly (Industrial Machine Fault Diagnosis)

| Mo hinh | Tac vu | Mau danh gia | Accuracy | Precision | Recall | F1-Score | F2-Score | FAR (%) | Lead Time (phut) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Isolation Forest (Baseline)** | Lami-04 Early Anomaly Detection | 360.0 | 0.4222 | 0.75 | 0.0143 | 0.028 | **0.0178** | 0.67 | **12.5** | 0.113 |
| **Deep AutoEncoder (1D-CNN)** | Lami-04 Early Anomaly Detection | 360.0 | 0.4778 | 0.9583 | 0.1095 | 0.1966 | **0.1331** | 0.67 | **21.0** | 0.033 |
| **MMLDA (Local Domain Adaptation)** | Cutting Fleet Fault Diagnosis (11 Machines) | 450.0 | 0.3422 | 1.0 | 0.0133 | 0.0263 | **0.0166** | 0.0 | **28.5** | 0.023 |


## 2. Phan he Khong gian mang (MQTT IoT IDS)

| Mo hinh | Tap kiem thu | Accuracy | F1-Macro | F1-Weighted | Attack Detection (MQTT BF) | Latency (ms) | Throughput (flows/s) |
|---|---|---|---|---|---|---|---|
| **Gaussian Naive Bayes (Baseline)** | 125.0 | 0.2 | **0.2** | 0.2 | 0.0 | 0.005 | 203549.9 |
| **Decision Tree** | 250.0 | 1.0 | **1.0** | 1.0 | 1.0 | 0.002 | 554939.0 |
| **Random Forest** | 250.0 | 1.0 | **1.0** | 1.0 | 1.0 | 0.244 | 4098.1 |


## 3. Phan he Tich hop (Cyber-Physical Dual-Tier Pipeline)

| Kien truc phoi hop | So chu ky test | Ty le chan tan cong (%) | Ty le thong luong hop le (%) | Tre Tier 1 Cyber (ms) | Tre Tier 2 Vat ly (ms) | Tong tre End-to-End (ms) | Tan so xu ly (Hz) |
|---|---|---|---|---|---|---|---|
| **MQTT IDS (RF) + MMLDA Multi-Scale** | 150.0 | **100.0** | 100.0 | 45.381 | 4.641 | **46.279** | 21.6 |


---
*Tai lieu tu dong sinh tu script `benchmark/run_all_benchmarks.py`.*