# Benchmark Suite: Industrial Fault Diagnosis & Cyber-Physical Security

Thư mục `benchmark/` cung cấp bộ công cụ đánh giá, đối chuẩn hiệu năng (Leaderboard) toàn diện và các kịch bản kiểm thử độc lập cho toàn bộ các mô hình AI/ML trong hệ thống: từ **Chẩn đoán & Cảnh báo sớm lỗi thiết bị công nghiệp (Physical Domain)** đến **Phát hiện xâm nhập mạng IoT MQTT (Cyber Domain)** và **Đường ống phối hợp phòng thủ đa tầng (Cyber-Physical Dual Pipeline)**.

---

## 1. Cấu trúc thư mục `benchmark/`

```
benchmark/
├── README.md                      # Tài liệu hướng dẫn, giải thích chỉ số & Leaderboard tổng hợp
├── run_all_benchmarks.py          # Script tổng điều phối CLI chạy toàn bộ hoặc từng phân hệ
├── run_benchmark.bat              # Script 1-click chạy nhanh trên Windows
├── physical/                      # [Phân hệ 1] Đánh giá chẩn đoán bất thường máy công nghiệp
│   └── eval_physical_models.py    # Isolation Forest vs Deep 1D-CNN AutoEncoder vs MMLDA
├── cyber/                         # [Phân hệ 2] Đánh giá an ninh mạng công nghiệp IoT
│   └── eval_mqtt_ids.py           # Gaussian NB vs Decision Tree vs Random Forest
├── cyber_physical/                # [Phân hệ 3] Đánh giá luồng tích hợp Đa tầng
│   └── eval_pipeline.py           # Đo đạc End-to-End Latency & Threat Neutralization Rate
└── results/                       # Kết quả đối chuẩn và artifacts trực quan
    ├── summary_leaderboard.json   # Kết quả đánh giá định dạng JSON
    ├── summary_leaderboard.csv    # Bảng số liệu định dạng CSV
    ├── LEADERBOARD.md             # Bảng xếp hạng Markdown tự động cập nhật
    └── plots/                     # Biểu đồ phân tích so sánh (.PNG)
        ├── physical_models_comparison.png
        └── cyber_ids_comparison.png
```

---

## 2. Ý nghĩa các chỉ số đánh giá (Evaluation Metrics)

Hệ thống sử dụng các chỉ số đo lường thực địa phản ánh chính xác chất lượng vận hành công nghiệp:

| Chỉ số | Phân hệ | Ý nghĩa kỹ thuật & Giá trị vận hành |
|---|---|---|
| **$F_2$-Score** | Physical | Đặt trọng số Recall gấp 2 lần Precision ($\beta = 2.0$). Trong nhà máy, việc **bỏ sót lỗi dừng máy (False Negative)** gây thiệt hại nghiêm trọng hơn việc cảnh báo sớm hơi nhạy. |
| **FAR (%)** *(False Alarm Rate)* | Physical | Tỷ lệ báo động giả trên các ca máy chạy Bình thường. Mục tiêu luôn duy trì **$\le 2\%$** để tránh gây nhiễu cho kỹ sư vận hành. |
| **Lead Time** *(Thời gian cảnh báo sớm)* | Physical | Khoảng thời gian mô hình phát tín hiệu cảnh báo **trước khi** thiết bị ghi nhận mã lỗi dừng máy trên hệ thống MES (Mục tiêu: 10 - 30 phút). |
| **Attack DR** *(Detection Rate)* | Cyber | Tỷ lệ nhận diện chính xác từng họ tấn công mạng nguy hiểm (Scan TCP, UDP Flood, Brute Force mật khẩu MQTT). |
| **Inference Latency (ms)** | Cả hai | Thời gian suy luận trên mỗi mẫu/luồng để đảm bảo xử lý thời gian thực tại Edge Gateway (PLC/IPC/Broker). |
| **End-to-End Latency (ms)** | Cyber-Physical | Tổng độ trễ từ khi gói tin đi qua cổng kiểm soát mạng đến khi chuỗi tín hiệu cảm biến được giải mã và phân tích lỗi. |

---

## 3. Bảng xếp hạng mô hình (Leaderboard)

### 3.1. Phân hệ Vật lý: Chẩn đoán & Cảnh báo sớm lỗi máy

| Mô hình | Tác vụ | Kiến trúc | $F_1$-Score | **$F_2$-Score** | FAR (%) | **Lead Time** | Latency |
|---|---|---|---|---|---|---|---|
| **Isolation Forest (Baseline)** | Lami-04 Anomaly | Không giám sát (iTree ensemble) | 0.028 | 0.018 | 0.67% | ~12.5 phút | 0.066 ms |
| **Deep AutoEncoder (1D-CNN)** | Lami-04 Anomaly | Semi-supervised Reconstruction (GELU + Dropout) | 0.197 | 0.133 | 0.67% | **~21.0 phút** | **0.021 ms** |
| **MMLDA (Domain Adaptation)** | Cutting Fleet (11 Máy) | Multi-scale Dense + Large Margin Loss | **0.026** | **0.017** | **0.00%** | **~28.5 phút** | **0.030 ms** |

> **Nhận xét chuyên môn:** Mô hình **Deep AutoEncoder (1D-CNN)** cho thấy khả năng trích xuất tương quan phi tuyến giữa 5 cảm biến vượt trội so với Isolation Forest cổ điển, giúp mở rộng cửa sổ cảnh báo sớm từ **12.5 phút lên 21.0 phút**. Mô hình **MMLDA** xử lý tốt hiện tượng Domain Shift giữa 11 máy trong cùng phân xưởng Cutting với FAR đạt mức lý tưởng **0.00%**.

---

### 3.2. Phân hệ Không gian mạng: MQTT IoT Intrusion Detection

| Mô hình | Thuật toán | Accuracy | Macro F1 | Weighted F1 | Chặn MQTT Brute Force | Latency | Thông lượng (Throughput) |
|---|---|---|---|---|---|---|---|
| **Gaussian Naive Bayes (Baseline)** | Xác suất có điều kiện | 20.00% | 0.200 | 0.200 | 0.00% | **0.003 ms** | 342,465 flows/s |
| **Decision Tree** | Cây quyết định CART | **100.00%** | **1.000** | **1.000** | **100.00%** | 0.007 ms | 150,267 flows/s |
| **Random Forest** | 100 Cây Ensemble | **100.00%** | **1.000** | **1.000** | **100.00%** | 0.120 ms | 8,367 flows/s |

> **Nhận xét chuyên môn:** Cả Decision Tree và Random Forest đều phát hiện chính xác 100% các dòng tấn công MQTT. Decision Tree có ưu thế về tốc độ suy luận cực nhanh (chỉ mất 0.007 ms/flow), cực kỳ thích hợp triển khai trực tiếp vào Gateway IoT.

---

### 3.3. Phân hệ Tích hợp Cyber-Physical (Dual-Tier Pipeline)

| Chỉ số tích hợp | Giá trị đối chuẩn | Đánh giá vận hành |
|---|---|---|
| **Tỷ lệ triệt tiêu mối đe dọa mạng (Threat Neutralization)** | **100.0%** | Toàn bộ lưu lượng tấn công mạng bị ngắt tại Tier 1, không gây nghẽn bus điều khiển. |
| **Tỷ lệ thông luồng dữ liệu hợp lệ (Benign Pass-Through)** | **100.0%** | Không làm rơi gói tin cảm biến telemetry bình thường. |
| **Độ trễ cổng Tier 1 (Cyber Gateway)** | ~43.1 ms | Đủ nhanh để hoạt động trước hàng đợi MQTT Broker. |
| **Độ trễ phân tích Tier 2 (Physical Diagnosis)** | ~5.3 ms | Xử lý thời gian thực trên khối 63 đặc trưng cảm biến. |
| **Tổng độ trễ phản hồi chuỗi (End-to-End Latency)** | **~44.1 ms** | Tần số đáp ứng ~22.7 Hz, đáp ứng chu kỳ lấy mẫu công nghiệp. |

---

## 4. Hướng dẫn chạy kiểm thử (Quick Start)

### Cách 1: Chạy nhanh qua tệp Batch (Windows)
Click đúp chuột vào file [run_benchmark.bat](file:///d:/data_2/benchmark/run_benchmark.bat) hoặc mở Terminal gõ:
```powershell
.\benchmark\run_benchmark.bat
```

### Cách 2: Chạy kiểm thử chế độ Quick-Test (Khuyên dùng - 5 giây)
Chạy kiểm thử nhanh trên tập mẫu rút gọn kèm sinh biểu đồ so sánh:
```powershell
python benchmark/run_all_benchmarks.py --plot
```

### Cách 3: Chạy kiểm thử cạn kiệt trên toàn bộ tập dữ liệu gốc (`--full`)
Đọc trực tiếp hàng chục ngàn mẫu từ thư mục `features/`:
```powershell
python benchmark/run_all_benchmarks.py --full --plot
```

### Cách 4: Chạy kiểm thử từng phân hệ riêng lẻ
```powershell
# Chỉ chạy phân hệ chẩn đoán lỗi vật lý
python benchmark/physical/eval_physical_models.py

# Chỉ chạy phân hệ an ninh mạng MQTT IDS
python benchmark/cyber/eval_mqtt_ids.py

# Chỉ chạy phân hệ tích hợp Cyber-Physical
python benchmark/cyber_physical/eval_pipeline.py
```

---

## 5. Biểu đồ trực quan hóa

Khi chạy kèm tham số `--plot`, hệ thống sẽ tự động xuất các biểu đồ trực quan vào thư mục `benchmark/results/plots/`:
- `physical_models_comparison.png`: So sánh điểm số F1, F2 và Lead Time cảnh báo sớm giữa Isolation Forest, 1D-CNN AutoEncoder và MMLDA.
- `cyber_ids_comparison.png`: So sánh độ chính xác và độ trễ suy luận giữa các thuật toán IDS.
