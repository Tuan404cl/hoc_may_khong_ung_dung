# DÀN Ý CHI TIẾT SLIDE THUYẾT TRÌNH (12 SLIDES)

**Chủ đề:** Hệ thống Điều khiển Máy tính Không chạm (Chuẩn *Point & Pinch* + *Windows Touchscreen Gestures*) sử dụng MediaPipe và 5 Thuật toán Học máy  
**Học phần:** Tương tác Người – Máy (HMI) – VNU-UET

---

## Slide 1: Trang Bìa (Title Slide)
- **Tên đề tài:** Nhận diện Cử chỉ Tay Thời gian thực cho Tương tác Máy tính Không chạm: Nâng cấp *Point & Pinch*, Cử chỉ Cảm ứng Chuẩn Windows và Đối sánh 5 Thuật toán Học máy.
- **Cơ sở khoa học:** Tái hiện & khắc phục toàn diện các điểm yếu của bài báo *Koduru et al. (IJCOPE 2026)*.
- **Thông tin nhóm / Sinh viên thực hiện.**

---

## Slide 2: Bước 1 – Xác định Bài toán & Phân tích Điểm yếu Hệ thống Cũ
- **Bài toán thực tế:** Tương tác máy tính không chạm trong y tế vô trùng, thuyết trình từ xa, hỗ trợ vận động tinh.
- **6 Điểm yếu được phát hiện qua kiểm thử phiên bản gốc:**
  1. Xung đột ML giữa 2 lớp cùng có `[0,1,1,0,0]` (`MOVE_POSE` vs `TWO_FINGER_CLOSE_POSE` – chiếm >80% lỗi của KNN/SVM).
  2. Công thái học kém: giơ ngón giữa để Click trái, giơ ngón út để Double Click, mỏi và phản cảm.
  3. Trôi con trỏ khi gập ngón để Click.
  4. Lỗi *Midas Touch* (không có tư thế nghỉ tay an toàn).
  5. Lỗi lật gương / tay trái-phải của công thức ngón cái gốc ($x_4 > x_3$).
  6. Bắn nhầm lệnh từ 1 frame nhiễu (Ghost Click) & đếm giữ tĩnh bằng frame khi chạy 100 FPS.

---

## Slide 3: Bộ Cử chỉ Mới Chuẩn Tự nhiên (*Point & Pinch + Windows Touch*)
- **10 Lớp Tư thế Tĩnh Trực giao (Thao tác chuẩn Apple Vision Pro + Windows Touchpad):**
  - **Chỉ 1 ngón trỏ (`POINT_POSE`):** Di chuyển chuột + Tự đóng băng con trỏ (`Pinch Freeze`) khi chuẩn bị bấm.
  - **Chụm Cái-Trỏ (`PINCH_INDEX`):** Chạm nhanh & thả = **Left Click** \| Chụm 2 lần = **Double Click** \| Giữ $\ge 0.35\text{s}$ = **Drag & Drop**.
  - **Chụm Cái-Giữa (`PINCH_MIDDLE`):** **Right Click**.
  - **Giơ 2 ngón Trỏ-Giữa (`TWO_FINGER_SCROLL`):** Cuộn Dọc & Ngang như Touchpad (gộp thành 1 lớp ML duy nhất).
  - **Chữ L (`ZOOM_L`):** Phóng to / Thu nhỏ (`Zoom In / Out`).
  - **Cử chỉ Windows 3 & 4 ngón:** Vuốt 3 ngón (`Win+Tab`, `Win+D`, `Alt+Tab`, giữ $0.85\text{s}$ = `Win+S`), Vuốt 4 ngón (`Virtual Desktop`, `Snap Window`, giữ $0.85\text{s}$ = `Win+A`).
  - **Xòe 5 ngón (`OPEN_PALM`):** Âm lượng + `Widgets (Win+W)`.
  - **Shaka Cái+Út (`SHAKA_BRIGHTNESS`):** Tăng/Giảm độ sáng (tách hẳn khỏi Zoom).
  - **Nắm đấm (`FIST_IDLE`):** Tư thế **Nghỉ tay An toàn (`CLUTCH_IDLE`)**.

---

## Slide 4: Bước 2 – Thu thập & Chuẩn bị Bộ Dữ liệu 10 Lớp
- **Nguồn dữ liệu thực tế trên mạng:**
  - Tải trực tiếp bộ dữ liệu tọa độ 21 điểm mốc MediaPipe công khai trên GitHub (`kinivi/hand-gesture-recognition-mediapipe` – **4,787 mẫu tay thật**).
- **Chuẩn hóa & Đối xứng Tay Trái / Tay Phải (`data/hand_gestures_dataset.csv`):**
  - Quy mô: **4,500 mẫu** (`450 mẫu/lớp` $\times$ `10 lớp`), chia **80% Train (3,600 mẫu)** và **20% Test (900 mẫu)**.
  - Căn chỉnh chiều lòng bàn tay trước khi tổng hợp & lật gương 50% dữ liệu để đảm bảo 100% bất biến tay trái/phải.

---

## Slide 5: Bước 3 – Trích chọn 54 Đặc trưng Hình học Bất biến Gương
- **Vì sao nâng từ 52 lên 54 đặc trưng?**
  1. **42 tọa độ chuẩn hóa hướng (`chirality_sign`):** Dịch gốc về cổ tay (`Landmark 0`), chuẩn hóa chiều ngang theo dấu `LM5_x - LM17_x` và chia cho bán kính bàn tay $S$.
  2. **5 trạng thái ngón tay bất biến góc quay & lật gương:** Ngón cái dùng phép chiếu lên vector trục ngang lòng bàn tay $\mathbf{u} = \frac{\text{LM}_5 - \text{LM}_{17}}{\|\text{LM}_5 - \text{LM}_{17}\|}$ và tỷ lệ $d(4,17)/d(3,17) > 1.06$.
  3. **7 khoảng cách & tỷ lệ hình học:** Bổ sung `dist_thumb_middle_norm` (nhận diện Click Phải) và `thumb_outward_ratio`.

---

## Slide 6: Bước 4 – Kiến trúc 5 Module Học máy Độc lập
- Thiết kế tách biệt 5 thuật toán thành 5 module độc lập trong `src/models/`:
  1. `svm_model.py` (`SVM RBF Kernel`, $C=3.0$)
  2. `knn_model.py` (`KNN`, $k=5$, distance-weighted)
  3. `logistic_regression_model.py` (`Multinomial Logistic Regression L2`)
  4. `decision_tree_model.py` (`CART`, `max_depth=12`, Gini)
  5. `naive_bayes_model.py` (`Gaussian Naive Bayes`, `var_smoothing=0.08`)
- Chuyển đổi nóng tức thời giữa 5 mô hình ngay trên **Pinned Floating Debug UI** bằng phím `1`–`5` hoặc nút bấm.

---

## Slide 7: Bước 5 – Kết quả Đánh giá & Bước Nhảy Vọt Trước vs. Sau
- **Chèn hình:** `results/model_comparison_chart.png`
- **Bảng so sánh thực nghiệm trên 900 mẫu Test:**
  - **SVM (RBF):** **98.89%** (`F1 = 98.89%`, tăng từ 92.16%, giảm 85% lỗi)
  - **KNN ($k=5$):** **98.67%** (`F1 = 98.67%`, tăng từ 90.68%, giảm 85% lỗi)
  - **Logistic Regression:** **97.89%** (`F1 = 97.89%`, `Infer = 0.0006 ms`, tăng từ 95.45%)
  - **Decision Tree:** **95.89%** (`F1 = 95.89%`, tăng từ 90.23%)
  - **Naive Bayes:** **88.67%** (`F1 = 88.39%`, tăng từ 87.61%)

---

## Slide 8: Bước 5 (Tiếp) – Đối sánh Ma trận Nhầm lẫn (Confusion Matrices)
- **Chèn hình:** `results/all_confusion_matrices.png`
- Đường chéo chính của `SVM`, `KNN`, `Logistic Regression` gần như tuyệt đối (chỉ còn 10–19/900 lỗi). Cặp xung đột 76 lỗi trước kia đã biến mất hoàn toàn.

---

## Slide 9: Bước 6 – Phân tích Các Trường hợp Dự đoán Sai (Error Analysis)
1. **Nhầm lẫn `FIST_IDLE` (9) vs. `PINCH_INDEX` (1) (chủ yếu ở GNB – 35 lỗi):**
   - Cả hai cùng có `[0,0,0,0,0]`, nhưng `SVM/KNN` tách rõ nhờ `dist_thumb_index_norm` ($0.10$ vs $0.47$), còn `GNB` bị ảnh hưởng bởi giả định độc lập trên 42 tọa độ xương.
2. **Nhầm lẫn `POINT` (0) vs. `PINCH_MIDDLE` (2) (31 lỗi ở GNB, 0 lỗi ở SVM):**
   - Cả hai cùng giữ ngón trỏ duỗi `[0,1,0,0,0]`, khác nhau chủ yếu ở khoảng cách `d(Cái, Giữa)`.
3. **Nhóm đa ngón (`OPEN_PALM` vs `FOUR_FINGER`, 2–4 lỗi):** Do các mẫu chuyển tiếp tư thế (5.5%).

---

## Slide 10: Các Cơ chế Bảo vệ Thời gian thực (Anti-Ghost & High-FPS)
- **MediaPipe VIDEO Mode + ThreadedCameraCapture:** Tách luồng đọc webcam `MJPG` khỏi giao diện, xử lý khung hình ở `256x256` giúp đạt **105+ FPS** trên CPU laptop.
- **Temporal Pose Stabilizer (`CONFIRM_FRAMES = 3`):** Chặn 100% nhiễu chuyển tiếp 1–2 frame (Ghost Click).
- **Pinch-Approach Cursor Freeze:** Khi $d(4,8)/L_{\text{palm}} < 0.45$, tự động tăng làm mượt EMA lên `4x` ($\alpha=32$) chống trượt nút bấm.
- **Bộ đếm giữ theo giây (`0.35s` Drag, `0.85s` Win Search/Action Center):** Hoạt động nhất quán ở mọi mức FPS.

---

## Slide 11: Minh họa Demo Giao diện Debug Ghim Nổi & Kiểm thử Tự động
- **Chèn hình:** `results/demo_hud_preview.png`
- Trình diễn **Pinned Debug Overlay UI**: hiển thị trực tiếp camera, khung xương, đường đo khoảng cách `pIdx = d(Cái-Trỏ)` & `pMid = d(Cái-Giữa)`, trạng thái 5 ngón tay, lệnh đang xuất ra và nhật ký debug.

---

## Slide 12: Kết luận & Hướng phát triển
- Giải quyết toàn diện cả 6 điểm yếu về ML lẫn công thái học tương tác (HMI).
- 4/5 mô hình đạt **95.89% – 98.89%** trên dữ liệu đa tay trái/phải, toàn bộ unit tests & kiểm thử hành vi vượt qua 100%.
- Hướng phát triển: Tinh chỉnh ngưỡng cá nhân hóa theo bàn tay từng người và tích hợp thêm cử chỉ vẽ/ghi chú trực tiếp trên màn hình thuyết trình.
