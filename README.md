# Touchless Virtual Mouse & Windows Touchscreen Gestures (MediaPipe + 5 Independent ML Models)

Dự án tái hiện và nâng cấp toàn diện bài báo khoa học **"Real-Time Hand Gesture Recognition for Touchless Computer Interaction: A MediaPipe-Based Virtual Mouse Implementation" (Koduru et al., IJCOPE 2026)** lên **Hệ thống Cử chỉ Tự nhiên Chuẩn HMI (Point & Pinch + Windows Precision Touchpad)**, tích hợp **5 Module Thuật toán Học máy Độc lập** (`KNN`, `Decision Tree`, `Naïve Bayes`, `Logistic Regression`, `SVM`), **bộ ổn định khung hình chống rung (3-frame Temporal Stabilizer)**, **đóng băng con trỏ khi chuẩn bị chụm (Pinch-Approach Freeze)** và **giao diện Debug ghim nổi tốc độ cao (Pinned Debug Overlay UI)**.

---

## 1. Cấu trúc Dự án

```text
Touch_less_screen_cuoi_ki/
├── requirements.txt                  # Các thư viện Python cần thiết
├── README.md                         # Hướng dẫn cài đặt và chạy toàn bộ hệ thống
├── data/
│   ├── raw_keypoint_github.csv       # Dữ liệu gốc tải trực tiếp từ GitHub (kinivi/Kazuhito00)
│   └── hand_gestures_dataset.csv     # Bộ dữ liệu 4,500 mẫu x 54 đặc trưng (10 lớp tư thế trực giao)
├── models/                           # Trọng số đã huấn luyện riêng của 5 mô hình ML & hand_landmarker.task
│   ├── knn_model.pkl
│   ├── dt_model.pkl
│   ├── nb_model.pkl
│   ├── lr_model.pkl
│   └── svm_model.pkl
├── results/                          # Kết quả thực nghiệm, biểu đồ & phân tích lỗi riêng từng mô hình
│   ├── knn/                          # metrics.json, confusion_matrix.png, error_analysis.md
│   ├── dt/                           # metrics.json, confusion_matrix.png, error_analysis.md
│   ├── nb/                           # metrics.json, confusion_matrix.png, error_analysis.md
│   ├── lr/                           # metrics.json, confusion_matrix.png, error_analysis.md
│   ├── svm/                          # metrics.json, confusion_matrix.png, error_analysis.md
│   ├── model_comparison.csv          # Bảng tổng hợp Accuracy, Precision, Recall, F1 của 5 mô hình
│   ├── model_comparison_chart.png    # Biểu đồ cột so sánh 5 mô hình
│   ├── all_confusion_matrices.png    # Hình ghép 5 Confusion Matrix của 5 mô hình
│   └── demo_hud_preview.png          # Hình minh họa giao diện HUD của hệ thống
├── docs/
│   ├── BAO_CAO_THUC_NGHIEM.md        # Báo cáo thực nghiệm chi tiết 6 bước học phần (VNU-UET HMI)
│   └── SLIDE_OUTLINE.md              # Nội dung chi tiết để làm Slide thuyết trình
├── src/
│   ├── config.py                     # Cấu hình 10 lớp tư thế trực giao, ngưỡng ổn định & hằng số
│   ├── features.py                   # Bước 3: Trích chọn 54 đặc trưng bất biến tay trái/phải & lật gương
│   ├── download_dataset.py           # Bước 2a: Tải dataset thực tế từ GitHub & tạo dataset 10 lớp (4,500 mẫu)
│   ├── collect_data.py               # Bước 2b: Công cụ thu thập thêm mẫu qua Webcam (phím 0..9)
│   ├── hand_tracker.py               # Camera đa luồng (ThreadedCameraCapture) & MediaPipe VIDEO mode tốc độ cao
│   ├── models/                       # Bước 4 & 5: 5 MODULE MÔ HÌNH HỌC MÁY ĐỘC LẬP
│   │   ├── base_evaluator.py         # Engine đánh giá & phân tích lỗi dùng chung
│   │   ├── knn_model.py              # Module 1: K-Nearest Neighbors (KNN)
│   │   ├── decision_tree_model.py    # Module 2: Decision Tree (CART)
│   │   ├── naive_bayes_model.py      # Module 3: Gaussian Naive Bayes
│   │   ├── logistic_regression_model.py # Module 4: Logistic Regression
│   │   └── svm_model.py              # Module 5: Support Vector Machine (SVM)
│   ├── evaluate_all.py               # Chạy kiểm tra cả 5 module & vẽ biểu đồ đối sánh
│   ├── controller.py                 # Bước 6a: Temporal Stabilizer, Pinch Click/Drag & Windows Touch Actions
│   ├── floating_ui.py                # Giao diện Debug ghim nổi luôn hiện trên cùng Windows
│   └── virtual_mouse.py              # Bước 6b: Ứng dụng Webcam Real-time (chuyển đổi nóng 5 mô hình)
└── tests/                            # Bộ Unit Tests tự động kiểm chứng toàn bộ hệ thống
    ├── test_features.py
    ├── test_five_models.py
    ├── test_controller.py
    ├── test_floating_ui.py
    └── test_hand_tracker.py
```

---

## 2. Hướng dẫn Cài đặt & Chạy Từng Bước

### Bước 1: Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

### Bước 2: Tải dữ liệu thực tế từ GitHub & Chuẩn bị Dataset 10 lớp (54 đặc trưng)
```powershell
python -m src.download_dataset
```
*(Nếu muốn tự thu thập thêm dữ liệu bằng Webcam của bạn: `python -m src.collect_data`)*

### Bước 3: Huấn luyện & Kiểm tra Độc lập Từng Mô hình Học máy
Bạn có thể chạy độc lập từng module mô hình để kiểm tra riêng:
```powershell
python -m src.models.knn_model
python -m src.models.decision_tree_model
python -m src.models.naive_bayes_model
python -m src.models.logistic_regression_model
python -m src.models.svm_model
```
Hoặc chạy tổng hợp cả 5 module để xuất bảng và biểu đồ đối sánh:
```powershell
python -m src.evaluate_all
```

### Bước 4: Chạy Ứng dụng Virtual Mouse & Windows Touchscreen (Real-Time Webcam)
Khởi chạy giao diện **Ghim nổi Debug UI** với bất kỳ mô hình nào bạn muốn kiểm tra (`knn`, `dt`, `nb`, `lr`, `svm`):
```powershell
# Chạy ở chế độ xem thử an toàn (chỉ debug lệnh, không tác động chuột/phím thật):
python -m src.virtual_mouse --model svm --dry-run

# Chạy điều khiển chuột & cử chỉ Windows thật với mô hình SVM:
python -m src.virtual_mouse --model svm

# Hoặc dùng cửa sổ OpenCV cổ điển thay cho Debug UI:
python -m src.virtual_mouse --model svm --classic-cv --dry-run
```
**Phím tắt trực tiếp khi đang mở cửa sổ:**
- Bấm phím `1`: Chuyển sang kiểm tra trực tiếp mô hình **KNN**
- Bấm phím `2`: Chuyển sang kiểm tra trực tiếp mô hình **Decision Tree**
- Bấm phím `3`: Chuyển sang kiểm tra trực tiếp mô hình **Naïve Bayes**
- Bấm phím `4`: Chuyển sang kiểm tra trực tiếp mô hình **Logistic Regression**
- Bấm phím `5`: Chuyển sang kiểm tra trực tiếp mô hình **SVM**
- Bấm phím `d`: Bật / Tắt chế độ `Dry-Run` (chỉ hiển thị lệnh hoặc điều khiển hệ thống thật)
- Bấm phím `q` hoặc `ESC`: Thoát chương trình.

### Bước 5: Chạy Kiểm thử Tự động (Unit Tests)
```powershell
python -m pytest tests/ -v
```
