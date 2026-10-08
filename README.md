# Industrial Machine Fault Diagnosis & Anomaly Detection (model_2)

Dự án nghiên cứu, huấn luyện và đánh giá các mô hình học máy (Machine Learning) và học sâu (Deep Learning) cho bài toán **Chẩn đoán & Cảnh báo sớm bất thường thiết bị công nghiệp (Early Anomaly Detection)**.

## 1. Mục tiêu bài toán
- **Mục tiêu cốt lõi:** Phát hiện sớm các bất thường vận hành (Early Anomaly Detection) trước khi thiết bị gặp sự cố dừng máy (Downtime) hoặc phát sinh lỗi cơ khí nghiêm trọng (ví dụ: bạc đạn con lăn `EE_13`, lệch film `EE_09`...).
- **Phương pháp học:** Bán giám sát (Semi-supervised / One-Class Anomaly Detection) — mô hình chỉ học phân phối của dữ liệu máy chạy Bình thường (Normal State) và phát hiện các sai lệch tái tạo/outlier score.
- **Chiến lược tiếp cận 2 tầng (Two-tier Approach):**
  - **Tầng 1 (Feature-based ML):** Trích xuất đặc trưng thống kê & phổ tần số (RMS, Kurtosis, Crest Factor, FFT) + Isolation Forest / One-Class SVM.
  - **Tầng 2 (Deep Multivariate Time-Series):** 1D-CNN AutoEncoder / TCN AutoEncoder học trực tiếp trên cửa sổ tín hiệu đa biến (Multivariate Time Series).
- **Ground Truth & Đánh giá:** Đối chiếu với nhật ký lỗi MES (`equipment_errors`, `lot_history_error_logs`, `lot_history_daily_stop_seconds`) với cửa sổ cảnh báo sớm (Lead Time) từ 5 đến 30 phút, tối ưu hóa theo chỉ số **$F_2$-score** và tỷ lệ cảnh báo giả (**FAR**).

## 2. Cấu trúc thư mục
```
├── docs/                      # Tài liệu tổng quan bài toán và nghiên cứu
│   ├── industrial-fault-diagnosis-overview.md
│   ├── paper-analysis-and-mes-integration.md
│   └── huong-dan-tinh-do-tin-cay-tinh-nang.md
├── src/                       # Mã nguồn tiền xử lý, trích xuất đặc trưng & mô hình
├── .gitignore                 # Bỏ qua các file dữ liệu nặng (>100MB)
└── README.md
```

> **Lưu ý:** Thư mục dữ liệu `raw/` và `csv_raw_chunks/` có kích thước lớn (> 2GB) nên được quản lý cục bộ và bỏ qua trong Git theo chính sách dung lượng của GitHub.
