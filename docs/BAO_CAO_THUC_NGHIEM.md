# BÁO CÁO THỰC NGHIỆM DỰ ÁN HỌC PHẦN TƯƠNG TÁC NGƯỜI - MÁY (HMI)

**Đề tài:** Nhận diện Cử chỉ Tay Thời gian thực cho Tương tác Máy tính Không chạm – Chuẩn Cử chỉ Tự nhiên *Point & Pinch* và *Windows Precision Touchpad* dựa trên MediaPipe & 5 Thuật toán Học máy  
**Trường:** Đại học Công nghệ – Đại học Quốc gia Hà Nội (VNU-UET)  
**Tài liệu gốc đối chiếu:** *Real-Time Hand Gesture Recognition for Touchless Computer Interaction: A MediaPipe-Based Virtual Mouse Implementation (Koduru et al., IJCOPE 2026)*

---

## BƯỚC 1: XÁC ĐỊNH BÀI TOÁN VÀ MỤC TIÊU ĐẦU RA

### 1.1. Đặt vấn đề & Cải tiến từ Bài báo Gốc
Các thiết bị đầu vào truyền thống (chuột vật lý, touchpad, màn hình cảm ứng chạm trực tiếp) bộc lộ hạn chế lớn trong môi trường vô trùng y tế, thuyết trình từ xa, ki-ốt công cộng hoặc hỗ trợ người khuyết tật vận động tinh. Bài báo của Koduru et al. (2026) đã đề xuất Virtual Mouse bằng MediaPipe nhưng tồn tại **6 điểm yếu cốt lõi** khi thử nghiệm thực tế:
1. **Xung đột lớp trong ML:** Giơ chữ V (`MOVE_POSE`) và 2 ngón khép (`TWO_FINGER_CLOSE_POSE`) cùng có cấu hình `[0,1,1,0,0]`, chỉ khác khoảng cách quanh ngưỡng $40\text{px}$, gây hơn 80–90% tổng số lỗi của KNN/SVM/NB.
2. **Thiếu công thái học:** Giơ riêng ngón giữa để Click trái hay chỉ giơ ngón út để Double Click rất mỏi cơ và phản cảm khi thuyết trình.
3. **Trôi chuột khi Click:** Đổi từ chữ V sang gập ngón trỏ làm đầu ngón tay bị giật khỏi nút bấm.
4. **Hội chứng chạm nhầm (*Midas Touch*):** Thiếu tư thế nghỉ tay an toàn (*Safe Clutch*), khiến người dùng không thể thu tay mà không bắn nhầm lệnh.
5. **Lỗi lật gương / tay trái-phải ở ngón cái:** Luật `x_tip(4) > x_ip(3)` sai hoàn toàn khi dùng tay trái hoặc camera lật gương.
6. **Kích hoạt nhiễu 1 khung hình & đếm frame:** Thiếu lọc ổn định chuyển tiếp (*Ghost Click*) và đếm tĩnh bằng frame khiến giữ tay 3–4 ngón nổ `Win+S`/`Win+A` quá sớm ở camera tốc độ cao (~100 FPS).

Trong phiên bản nâng cấp này, chúng tôi thiết kế lại toàn bộ bộ từ vựng thành **10 Lớp Tư thế Trực giao Chuẩn Tự nhiên (*Point & Pinch + Windows Touch*)**:
- Tách rời hình thái ngón tay để **triệt tiêu cặp lớp xung đột `[0,1,1,0,0]`**, giúp độ chính xác của `SVM`, `KNN`, `Logistic Regression`, `Decision Tree` vọt lên **95.89% – 98.89%**.
- Áp dụng tư thế **Chỉ 1 ngón trỏ** để di chuột, **Chụm Cái-Trỏ** (chạm nhanh = Click Trái, chụm 2 lần = Double Click, giữ $\ge 0.35\text{s}$ = Kéo thả Drag & Drop), **Chụm Cái-Giữa** = Click Phải, **2 Ngón Trỏ-Giữa** = Cuộn trang 4 chiều như touchpad laptop, **Chữ L** = Pinch Zoom, **Shaka (Cái+Út)** = Độ sáng, và **Nắm đấm** làm tư thế **Tạm dừng an toàn (`CLUTCH_IDLE`)**.

### 1.2. Định nghĩa Bài toán Học máy
- **Đầu vào ($\mathbf{x} \in \mathbb{R}^{54}$):** Vector **54 đặc trưng bất biến với vị trí, tỷ lệ và tay trái/phải – lật gương**, trích xuất từ 21 điểm mốc (landmarks) bàn tay do MediaPipe HandLandmarker (`RunningMode.VIDEO`) phát hiện.
- **Đầu ra của Mô hình Học máy ($y \in \{0, 1, \dots, 9\}$):** Nhãn của **10 lớp tư thế bàn tay tĩnh trực giao**.
- **Đầu ra Hệ thống (`System Action`):** Qua bộ ổn định **Temporal Pose Stabilizer (3 khung hình liên tiếp)**, bộ đóng băng con trỏ khi chuẩn bị chụm (**Pinch-Approach Cursor Freeze**, $\alpha \times 4$), làm mượt **EMA ($\alpha=8$)**, chống rung **Debouncing ($0.4\text{s}$)** và bộ đếm thời gian thực (`time.monotonic()`) để điều khiển chuột & cử chỉ đa nhiệm Windows 10/11.

### 1.3. Bảng Ánh xạ 10 Lớp Tư thế Học máy sang Cử chỉ Tự nhiên (*Point & Pinch + Windows Touch*)

| Mã Lớp ($y$) | Tên Lớp Tư thế (`Pose Class`) | Mẫu Ngón tay `[Cái, Trỏ, Giữa, Áp, Út]` | Hành động Điều khiển Hệ thống |
| :---: | :--- | :---: | :--- |
| **0** | `POINT_POSE` | `[0, 1, 0, 0, 0]` (1 ngón trỏ) | **Di chuyển con trỏ chuột** mượt theo EMA ($\alpha=8$) + *Tự động đóng băng con trỏ 4x khi ngón cái tiến sát để chuẩn bị Click* |
| **1** | `PINCH_INDEX_POSE` | Chụm đầu Cái + Trỏ (3 ngón sau gập) | **Chụm nhanh & thả ($<0.35\text{s}$):** Click Trái (`LEFT_CLICK`)<br>**Chụm 2 lần nhanh ($\le 0.45\text{s}$):** `DOUBLE_CLICK`<br>**Chụm & Giữ ($\ge 0.35\text{s}$):** Kéo thả (`DRAG_START` / `DRAG_MOVE` / `DRAG_RELEASE`) |
| **2** | `PINCH_MIDDLE_POSE` | Chụm Cái + Giữa (ngón trỏ vẫn duỗi) | **Click Chuột Phải (`RIGHT_CLICK`)** – khóa kích hoạt 1 lần cho mỗi nhịp chụm |
| **3** | `TWO_FINGER_SCROLL_POSE` | `[0, 1, 1, 0, 0]` (2 ngón Trỏ + Giữa) | **Cuộn Dọc** (`SCROLL_VERTICAL`) & **Cuộn Ngang** (`SCROLL_HORIZONTAL`) như Touchpad Windows |
| **4** | `ZOOM_L_POSE` | `[1, 1, 0, 0, 0]` (Chữ L: Cái + Trỏ mở) | **Khép/Mở khoảng cách Cái-Trỏ:** `ZOOM_IN` (`Ctrl`+`+`) / `ZOOM_OUT` (`Ctrl`+`-`) |
| **5** | `THREE_FINGER_POSE` | `[0, 1, 1, 1, 0]` (3 ngón) | **Vuốt Lên:** Task View (`Win+Tab`) \| **Vuốt Xuống:** Show Desktop (`Win+D`)<br>**Vuốt Trái/Phải:** Chuyển cửa sổ (`Alt+Tab`) \| **Giữ tĩnh $0.85\text{s}$:** Search (`Win+S`) |
| **6** | `FOUR_FINGER_POSE` | `[0, 1, 1, 1, 1]` (4 ngón) | **Vuốt Trái/Phải:** Virtual Desktop (`Win+Ctrl+Left/Right`)<br>**Vuốt Lên/Xuống:** Snap Maximize/Minimize (`Win+Up/Down`) \| **Giữ tĩnh $0.85\text{s}$:** Action Center (`Win+A`) |
| **7** | `OPEN_PALM_POSE` | `[1, 1, 1, 1, 1]` (Xòe 5 ngón) | **Đưa Lên/Xuống:** Tăng/Giảm Âm lượng \| **Vuốt sang Phải:** Mở bảng Widgets (`Win+W`) |
| **8** | `SHAKA_BRIGHTNESS_POSE` | `[1, 0, 0, 0, 1]` (Shaka: Cái + Út) | **Đưa Lên/Xuống:** Tăng/Giảm Độ sáng màn hình $\pm 10\%$ *(Tách biệt 100% khỏi Zoom)* |
| **9** | `FIST_IDLE_POSE` | `[0, 0, 0, 0, 0]` (Nắm đấm) | **Tạm dừng / Nghỉ tay an toàn (`CLUTCH_IDLE`)** – khóa mọi lệnh chuột để đưa tay về vị trí thoải mái |

---

## BƯỚC 2: THU THẬP VÀ LỰA CHỌN BỘ DỮ LIỆU

### 2.1. Nguồn dữ liệu trực tuyến kết hợp tái cấu trúc giải phẫu
Bộ dữ liệu được xây dựng bởi `src/download_dataset.py`:
1. **Dữ liệu gốc tải trực tiếp từ mạng (`data/raw_keypoint_github.csv`):** Tải trực tiếp bộ dữ liệu tọa độ 21 điểm mốc MediaPipe thực tế từ kho mã nguồn mở công khai [`kinivi/hand-gesture-recognition-mediapipe`](https://github.com/kinivi/hand-gesture-recognition-mediapipe) gồm **4,787 mẫu bàn tay thật**.
2. **Chuẩn hóa hướng tay trái/phải & Tái cấu trúc 10 lớp (`data/hand_gestures_dataset.csv`):** Căn chỉnh đồng bộ chiều lòng bàn tay trước khi tổng hợp từ khung xương thật, tái cấu trúc chính xác hình vòm chụm ngón (`PINCH_INDEX`, `PINCH_MIDDLE`), tư thế `Shaka`, `Chữ L`, và **lật gương ngẫu nhiên 50% số mẫu** để mô hình hoạt động đồng nhất cho cả tay trái và tay phải.
3. **Mô phỏng nhiễu thực tế:** Độ lệch xoay cổ tay $\theta \sim \mathcal{N}(0, 4.6^\circ)$, nhiễu cảm biến camera và **5.5% mẫu chuyển tiếp tư thế (transitional poses)**.

### 2.2. Quy mô và Phân bố Bộ dữ liệu
- **Tổng số mẫu:** `4,500` mẫu (`450` mẫu cho mỗi lớp trong 10 lớp).
- **Chia tập dữ liệu (Stratified Split):**
  - Tập huấn luyện (`Train set` – 80%): `3,600` mẫu (`360` mẫu/lớp).
  - Tập kiểm thử (`Test set` – 20%): `900` mẫu (`90` mẫu/lớp).

---

## BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU VÀ TRÍCH CHỌN 54 ĐẶC TRƯNG BẤT BIẾN

Module `src/features.py` khắc phục triệt để lỗi lật gương/tay trái-phải và trích xuất **vector 54 đặc trưng**:
1. **Nhóm 1 – 42 Tọa độ Tương đối Chuẩn hóa Hướng (`lm0_x .. lm20_y`):**
   Dịch gốc tọa độ về cổ tay (`Landmark 0`), chuẩn hóa dấu trục ngang theo chiều lòng bàn tay ($\text{chirality\_sign} = -1$ nếu $x_5 > x_{17}$ ngược lại $+1$), và chia cho bán kính bàn tay $S$:
   $$\hat{x}_i = \frac{(x_i - x_0) \cdot s_{\text{chirality}}}{S}, \quad \hat{y}_i = \frac{y_i - y_0}{S}$$
2. **Nhóm 2 – 5 Đặc trưng Duỗi Ngón tay Bất biến Gương (`thumb_up .. pinky_up`):**
   - **Ngón cái (bất biến lật gương & góc quay):** Chiếu vector đốt ngón cái $(\text{LM}_4 - \text{LM}_3)$ lên vector trục ngang lòng bàn tay $\mathbf{u} = \frac{\text{LM}_5 - \text{LM}_{17}}{\|\text{LM}_5 - \text{LM}_{17}\|}$, kết hợp tỷ lệ khoảng cách về gốc ngón út $\frac{d(4, 17)}{d(3, 17)} > 1.06$ và độ mở khỏi đốt ngón giữa $d(4, 10) > 0.45\,L_{\text{palm}}$.
   - **4 ngón còn lại:** $\text{Extended}_i = 1$ nếu $y_{\text{tip}_i} < y_{\text{pip}_i}$ và $d(0, \text{tip}_i) > d(0, \text{pip}_i)$.
3. **Nhóm 3 – 7 Khoảng cách & Tỷ lệ Hình học Chuẩn hóa (`dist_*`):**
   $d(4,8)/L_{\text{palm}}$ (Chụm Trỏ / Zoom L), $d(4,12)/L_{\text{palm}}$ (Chụm Giữa Click Phải), $d(8,12)/L_{\text{palm}}$, $d(12,16)/L_{\text{palm}}$, $d(16,20)/L_{\text{palm}}$, $d(4,20)/L_{\text{palm}}$ (Shaka), và $\text{thumb\_outward\_ratio} = d(4,17)/d(3,17)$.

Toàn bộ 54 đặc trưng được chuẩn hóa Z-score (`StandardScaler`) trong từng module độc lập.

---

## BƯỚC 4: HUẤN LUYỆN VÀ KIỂM TRA ĐỘC LẬP 05 MODULE MÔ HÌNH HỌC MÁY

Cả **5 thuật toán** chạy độc lập trong `src/models/`:
1. **KNN (`knn_model.py`):** $k = 5$, `weights='distance'`, khoảng cách Euclid trên 54 chiều.
2. **Decision Tree (`decision_tree_model.py`):** CART `gini`, `max_depth = 12`, `min_samples_split = 4`.
3. **Naïve Bayes (`naive_bayes_model.py`):** `GaussianNB(var_smoothing=0.08)` điều chuẩn phương sai cho các đặc trưng nhị phân.
4. **Logistic Regression (`logistic_regression_model.py`):** Đa lớp `Multinomial L2`, `C = 1.5`, `max_iter = 1000`.
5. **SVM (`svm_model.py`):** Nhân phi tuyến `RBF`, `C = 3.0`, `gamma = 'scale'`.

---

## BƯỚC 5: ĐÁNH GIÁ KẾT QUẢ & SO SÁNH TRƯỚC / SAU KHI ĐỔI BỘ CỬ CHỈ

### 5.1. Bảng Tổng hợp Đối sánh 5 Module Độc lập (Trên 900 mẫu Kiểm thử)

*(Trích xuất từ `results/model_comparison.csv`)*

| Mã Module | Tên Mô hình Học máy | Accuracy Cũ (11 lớp) | **Accuracy Mới (10 lớp)** | Precision Macro (%) | Recall Macro (%) | F1-Score Macro (%) | Suy luận (ms/mẫu) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SVM`** | **Support Vector Machine (RBF Kernel)** | 92.16% | **98.89%** *(-85% lỗi)* | **98.91%** | **98.89%** | **98.89%** | 0.1761 |
| **`KNN`** | **K-Nearest Neighbors (KNN, k=5)** | 90.68% | **98.67%** *(-85% lỗi)* | **98.68%** | **98.67%** | **98.67%** | 0.0445 |
| **`LR`** | **Logistic Regression (Multinomial L2)** | 95.45% | **97.89%** *(-52% lỗi)* | **97.93%** | **97.89%** | **97.89%** | **0.0006** |
| **`DT`** | **Decision Tree (CART, max_depth=12)** | 90.23% | **95.89%** *(-57% lỗi)* | **95.94%** | **95.89%** | **95.89%** | 0.0007 |
| **`NB`** | **Gaussian Naive Bayes (GNB)** | 87.61% | **88.67%** | **89.71%** | **88.67%** | **88.39%** | 0.0033 |

> **Nhận xét nâng cấp:**  
> Nhờ gộp 2 lớp xung đột `[0,1,1,0,0]` thành duy nhất `TWO_FINGER_SCROLL_POSE` và chuyển các cử chỉ chuột sang `POINT_POSE`, `PINCH_INDEX`, `PINCH_MIDDLE`, `SHAKA`, cả 4 mô hình chính (`SVM`, `KNN`, `LR`, `DT`) đều vượt mốc **95.8% – 98.89%** (vượt xa mức 94.0%–95.9% của bài báo Koduru et al. 2026), với thời gian suy luận cực nhỏ (**0.0006 – 0.1761 ms/mẫu**).

### 5.2. Biểu đồ So sánh & Ma trận Nhầm lẫn
- Biểu đồ so sánh 4 chỉ số: `results/model_comparison_chart.png`
- Ma trận nhầm lẫn tổng hợp 5 mô hình: `results/all_confusion_matrices.png`
- Ma trận nhầm lẫn chi tiết từng mô hình: `results/<knn|dt|nb|lr|svm>/confusion_matrix.png`

---

## BƯỚC 6: PHÂN TÍCH TRƯỜNG HỢP DỰ ĐOÁN SAI, HOÀN THIỆN DEMO VÀ ĐÁNH GIÁ HÀNH VI

### 6.1. Phân tích các Cặp Nhầm lẫn Còn lại (Trên 900 mẫu Test)
1. **Nhóm `FIST_IDLE_POSE` (9) vs `PINCH_INDEX_POSE` (1) (35 lỗi ở GNB; chỉ 1–3 lỗi ở KNN/LR/DT):**  
   Cả hai tư thế đều gập cả 4 ngón tay dài và khép ngón cái (`fingers_up = [0,0,0,0,0]`). Khi chỉ xét phân phối độc lập từng tọa độ trên không gian 54 chiều, `Gaussian Naive Bayes` chịu ảnh hưởng bởi độ biến thiên rộng của tập nắm đấm thật (`fist_rows`), trong khi `SVM`, `KNN`, `LR` tận dụng tốt tương quan không gian giữa `dist_thumb_index_norm` ($\approx 0.10$ khi chụm vs $\approx 0.47$ khi nắm) để đạt độ chính xác $>97\%$.
2. **Nhóm `POINT_POSE` (0) vs `PINCH_MIDDLE_POSE` (2) (31 lỗi ở GNB; 0 lỗi ở SVM, 1 ở KNN, 5 ở DT):**  
   Cả hai tư thế đều duỗi thẳng ngón trỏ (`[0,1,0,0,0]`), chủ yếu khác biệt ở `dist_thumb_middle_norm`. Đặc trưng đơn lẻ này bị pha loãng bởi 42 tọa độ xương khi giả định độc lập tuyến tính trong GNB bị vi phạm, nhưng được `SVM (RBF)` phân tách hoàn hảo với lề cực đại.
3. **Nhóm đa ngón Windows Touchscreen (`OPEN_PALM_POSE` vs `FOUR_FINGER_POSE`, 2–4 lỗi):**  
   Xảy ra ở một số mẫu có nhiễu chuyển tiếp (5.5%) khi ngón cái khép chưa hẳn vào trong hoặc mở chưa hết ra ngoài.

### 6.2. Khắc phục Triệt để 4 Lỗi Hành vi ở Tầng Điều khiển (`src/controller.py` & `src/floating_ui.py`)
1. **Chống Ghost Click chuyển tiếp:** Bộ ổn định `Temporal Pose Stabilizer` (`CONFIRM_FRAMES = 3`) yêu cầu cùng 1 pose xuất hiện 3 khung hình liên tiếp mới xác nhận chuyển trạng thái.
2. **Triệt tiêu trôi chuột khi bấm:** Khi phát hiện ngón cái tiến sát ngón trỏ (`d(4,8)/L_palm < 0.45`), `POINT_POSE` tự động tăng làm mượt EMA lên **4 lần** (`alpha * 4 = 32`) giúp con trỏ đứng vững trước khi chụm.
3. **Đo thời gian giữ theo đồng hồ thực (`time.monotonic()`):** Thay thế đếm frame bằng ngưỡng thời gian `0.85s` (`STATIC_HOLD_DURATION`) cho 3/4-ngón (`Win+S`, `Win+A`) và `0.35s` (`DRAG_HOLD_DURATION`) cho Pinch-to-Drag, hoạt động chính xác như nhau dù webcam chạy 30 FPS hay 100+ FPS.
4. **Cơ chế Nghỉ tay (`FIST_IDLE_POSE` -> `CLUTCH_IDLE`):** Nắm đấm trở thành khóa an toàn hoàn toàn không gắn bất kỳ phím tắt hay điều chỉnh độ sáng nào, tách rời Độ sáng sang cử chỉ `SHAKA_BRIGHTNESS_POSE` (`[1,0,0,0,1]`).
