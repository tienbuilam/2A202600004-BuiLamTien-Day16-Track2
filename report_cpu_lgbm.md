# Báo cáo Lab 16 — Phương án CPU + LightGBM

**Họ tên:** Bùi Lâm Tiến  
**Mã sinh viên:** 2A202600004  
**Instance:** `r5.2xlarge` (8 vCPU, 64GB RAM) — us-east-1  
**Dataset:** Credit Card Fraud Detection (284,807 giao dịch, 31 features)

---

## Lý do dùng CPU thay GPU

Tài khoản AWS mới bị giới hạn quota GPU mặc định ở mức 0 vCPU cho dòng G/VT instances (`g4dn.xlarge`). Yêu cầu tăng quota chưa được duyệt trong thời gian làm lab, do đó chuyển sang phương án CPU với instance `r5.2xlarge` — có chi phí tương đương (~$0.504/giờ so với $0.526/giờ của `g4dn.xlarge`) nhưng không cần quota đặc biệt.

---

## Kết quả Benchmark

| Metric | Kết quả |
|---|---|
| Thời gian load data | 1.63s |
| Thời gian training | 1.02s |
| Best iteration | 1 |
| AUC-ROC | 0.9415 |
| Accuracy | 0.9990 |
| F1-Score | 0.7421 |
| Precision | 0.6667 |
| Recall | 0.8367 |
| Inference latency (1 row) | 0.962ms |
| Inference throughput (1000 rows) | 1.185ms |

---

## Nhận xét

- **Training time cực nhanh (1.02s):** LightGBM với early stopping hội tụ chỉ sau 1 iteration, cho thấy gradient boosting rất hiệu quả trên tabular data so với Deep Learning vốn cần hàng giờ training ngay cả trên GPU.
- **AUC-ROC 0.9415:** Kết quả tốt cho bài toán fraud detection với dataset imbalanced nặng (chỉ 0.17% giao dịch là fraud).
- **Inference latency < 1ms/row:** Đủ nhanh cho production real-time scoring, không cần GPU acceleration cho loại mô hình này.
- **Kết luận:** Với bài toán ML truyền thống (tabular data + gradient boosting), CPU instance `r5.2xlarge` hoàn toàn đủ năng lực và tiết kiệm hơn so với GPU — GPU chỉ thực sự cần thiết khi chạy Deep Learning / LLM inference.
