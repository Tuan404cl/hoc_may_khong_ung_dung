# Phân tích Kết quả & Trường hợp Dự đoán sai - Gaussian Naive Bayes (GNB)

## 1. Chỉ số Đánh giá Tổng quát (Trên tập Kiểm thử 20%)
- **Accuracy:** `88.67%`
- **Precision (Macro):** `89.71%`
- **Recall (Macro):** `88.67%`
- **F1-Score (Macro):** `88.39%`
- **Tổng số mẫu kiểm thử:** `900` mẫu (`102` mẫu dự đoán sai)
- **Thời gian suy luận trung bình:** `0.0033 ms/frame`

## 2. Top các cặp Cử chỉ bị Nhầm lẫn nhiều nhất (Confusion Matrix Off-Diagonal)
| Hạng | Nhãn Thực tế (True) | Nhãn Dự đoán Nhầm (Predicted) | Số mẫu nhầm | Nguyên nhân Hình học & Giải thích |
| :--- | :--- | :--- | :---: | :--- |
| 1 | `FIST_IDLE_POSE` | `PINCH_INDEX_POSE` | **35** | Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp. |
| 2 | `POINT_POSE` | `PINCH_MIDDLE_POSE` | **31** | Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes). |
| 3 | `PINCH_INDEX_POSE` | `FIST_IDLE_POSE` | **6** | Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp. |
| 4 | `OPEN_PALM_POSE` | `FOUR_FINGER_POSE` | **4** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 5 | `PINCH_MIDDLE_POSE` | `POINT_POSE` | **3** | Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes). |
| 6 | `OPEN_PALM_POSE` | `FIST_IDLE_POSE` | **2** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 7 | `THREE_FINGER_POSE` | `FOUR_FINGER_POSE` | **2** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 8 | `POINT_POSE` | `FIST_IDLE_POSE` | **2** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |