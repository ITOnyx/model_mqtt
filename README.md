# Industrial Machine Fault Diagnosis & Cyber-Physical Security (model_2)

Dự án nghiên cứu, huấn luyện và đánh giá các mô hình học máy (Machine Learning) và học sâu (Deep Learning) cho bài toán **Chẩn đoán & Cảnh báo sớm bất thường thiết bị công nghiệp (Early Anomaly Detection)** kết hợp **Hệ thống phát hiện xâm nhập mạng công nghiệp (MQTT IoT IDS)**.

---

## 1. Mục tiêu bài toán
- **Mục tiêu cốt lõi:** Phát hiện sớm các bất thường vận hành (Early Anomaly Detection) trước khi thiết bị gặp sự cố dừng máy (Downtime) hoặc phát sinh lỗi cơ khí nghiêm trọng (bạc đạn con lăn `EE_13`, lệch film `EE_09`...).
- **Phương pháp học:** Bán giám sát (Semi-supervised / One-Class Anomaly Detection) — mô hình chỉ học phân phối của dữ liệu máy chạy Bình thường (Normal State) và phát hiện các sai lệch tái tạo/outlier score.
- **Chiến lược tiếp cận đa tầng (Multi-tier Approach):**
 - **Tầng 1 (Network Perimeter Gatekeeper):** Cây quyết định / Random Forest phân loại luồng mạng MQTT-IoT-IDS2020 nhằm loại bỏ tấn công brute-force, scan cổng trước khi vào bus điều khiển PLC/IPC.
 - **Tầng 2 (Feature-based ML):** Trích xuất đặc trưng thống kê & phổ tần số (RMS, Kurtosis, Crest Factor, FFT) + Isolation Forest.
 - **Tầng 3 (Deep Multivariate Time-Series & Fleet Domain Adaptation):** 1D-CNN AutoEncoder & MMLDA (Maximum Margin Local Domain Adaptation) học trên cửa sổ tín hiệu đa cảm biến trên dàn 11 máy Cutting và 10 máy Lamination.
- **Ground Truth & Đánh giá:** Đối chiếu với nhật ký lỗi MES (`equipment_errors`, `lot_history_error_logs`) với cửa sổ cảnh báo sớm (Lead Time) từ 10 đến 30 phút, tối ưu hóa theo chỉ số **$F_2$-score** và tỷ lệ cảnh báo giả (**FAR**).

---

## 2. Cấu trúc thư mục dự án

```
d:/data_2/
├── benchmark/                             # BỘ CÔNG CỤ ĐỐI CHUẨN HIỆU NĂNG & LEADERBOARD
│   ├── README.md                          # Tài liệu chi tiết, định nghĩa chỉ số & bảng Leaderboard
│   ├── run_all_benchmarks.py              # Script tổng điều phối CLI (--domain, --full, --plot)
│   ├── run_benchmark.bat                  # Script chạy nhanh 1-click trên Windows
│   ├── physical/                          # Đánh giá phân hệ chẩn đoán bất thường thiết bị
│   │   └── eval_physical_models.py        # Isolation Forest vs Deep AutoEncoder vs MMLDA
│   ├── cyber/                             # Đánh giá phân hệ phát hiện xâm nhập mạng MQTT IDS
│   │   └── eval_mqtt_ids.py               # Gaussian NB vs Decision Tree vs Random Forest
│   ├── cyber_physical/                    # Đánh giá phân hệ tích hợp phòng thủ đa tầng
│   │   └── eval_pipeline.py               # Đo đạc End-to-End Latency & Threat Neutralization Rate
│   └── results/                           # Kết quả xuất tự động & biểu đồ đối chuẩn
│       ├── LEADERBOARD.md                 # Bảng xếp hạng mô hình Markdown tự động cập nhật
│       ├── summary_leaderboard.json       # Kết quả đối chuẩn dạng JSON
│       ├── summary_leaderboard.csv        # Kết quả đối chuẩn dạng CSV
│       └── plots/                         # Biểu đồ so sánh trực quan (.PNG)
│           ├── physical_models_comparison.png
│           └── cyber_ids_comparison.png
├── models/                                # TRỌNG SỐ MÔ HÌNH ĐÃ HUẤN LUYỆN & METADATA
│   ├── deep_autoencoder_lami04.pt         # Model PyTorch Deep AutoEncoder cho máy Lami-04
│   ├── deep_autoencoder_metadata.joblib   # Scaler & ngưỡng threshold tái tạo lỗi Lami-04
│   ├── fleet_deep_autoencoder_lami.pt     # Model Fleet Deep AutoEncoder dàn máy Lami
│   ├── fleet_deep_autoencoder_cutting.pt  # Model Fleet Deep AutoEncoder dàn máy Cutting
│   ├── mmlda_fleet_fault_diagnosis.pt     # Model MMLDA Domain Adaptation (11 máy Cutting)
│   ├── mmlda_fleet_metadata.joblib        # Scaler & nhãn phân loại MMLDA
│   ├── isolation_forest_lami04.joblib     # Model Isolation Forest baseline + scaler
│   ├── mqtt_ids_decision_tree.joblib      # Model Decision Tree phân loại tấn công MQTT
│   └── mqtt_ids_random_forest.joblib      # Model Random Forest phân loại tấn công MQTT
├── src/                                   # MÃ NGUỒN TIỀN XỬ LÝ & MÔ HÌNH HÓA
│   ├── timescaledb_decoder.py             # Giải mã nén delta-delta dữ liệu TimescaleDB
│   ├── extract_features_sliding_window.py # Trích xuất 63 đặc trưng thời gian & tần số (RMS, FFT, Kurtosis)
│   ├── eda_telemetry_fault_linking.py     # Liên kết nhãn telemetry với lỗi MES
│   ├── train_isolation_forest_baseline.py # Huấn luyện mô hình cơ sở Isolation Forest
│   ├── train_deep_autoencoder.py          # Huấn luyện mô hình Deep 1D-CNN AutoEncoder
│   ├── fleet_lami_anomaly_detection.py    # Pipeline học máy mở rộng cụm máy Lamination
│   ├── fleet_cutting_anomaly_detection.py # Pipeline học máy mở rộng cụm máy Cutting
│   ├── mmlda_fault_diagnosis.py           # Huấn luyện MMLDA Domain Adaptation
│   ├── train_mqtt_ids_models.py           # Huấn luyện 6 mô hình MQTT IoT IDS
│   ├── simulate_cyber_physical_pipeline.py# Mô phỏng thời gian thực luồng Cyber-Physical
│   └── stream_inference_simulator.py      # Bộ giả lập suy luận luồng dữ liệu thời gian thực
├── .gitignore                             # Quản lý loại trừ file dữ liệu nặng (>100MB)
└── README.md                              # Tài liệu dự án
```

> **Lưu ý dữ liệu lớn:** Các thư mục dữ liệu thô `raw/`, `csv_raw_chunks/` và `features/` có dung lượng lớn (> 2GB) được lưu trữ cục bộ và quản lý theo `.gitignore`. Kết quả đối chuẩn và thông số hiệu năng của toàn bộ các mô hình được lưu trữ sẵn trong thư mục `benchmark/results/`.

---

## 3. Chạy kiểm thử & Đối chuẩn mô hình (Benchmark Quick Start)

Hệ thống hỗ trợ chạy kiểm thử tức thì không yêu cầu tải dữ liệu dung lượng lớn:

```powershell
# Chạy Quick-Test toàn bộ phân hệ và sinh biểu đồ (.PNG) trong 5 giây:
python benchmark/run_all_benchmarks.py --plot

# Chạy kiểm thử cạn kiệt trên toàn bộ tập dữ liệu gốc features/:
python benchmark/run_all_benchmarks.py --full --plot

# Chạy riêng từng phân hệ:
python benchmark/physical/eval_physical_models.py
python benchmark/cyber/eval_mqtt_ids.py
python benchmark/cyber_physical/eval_pipeline.py
```

*Chi tiết bảng xếp hạng Leaderboard, ý nghĩa chỉ số $F_2$, FAR và thời gian cảnh báo sớm (Lead Time) xem tại [benchmark/README.md](file:///d:/data_2/benchmark/README.md).*
