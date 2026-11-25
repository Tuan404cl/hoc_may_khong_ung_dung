# Phân tích Kết quả & Trường hợp Dự đoán sai - K-Nearest Neighbors (KNN, k=5)

## 1. Chỉ số Đánh giá Tổng quát (Trên tập Kiểm thử 20%)
- **Accuracy:** `98.67%`
- **Precision (Macro):** `98.68%`
- **Recall (Macro):** `98.67%`
- **F1-Score (Macro):** `98.67%`
- **Tổng số mẫu kiểm thử:** `900` mẫu (`12` mẫu dự đoán sai)
- **Thời gian suy luận trung bình:** `0.0445 ms/frame`

## 2. Top các cặp Cử chỉ bị Nhầm lẫn nhiều nhất (Confusion Matrix Off-Diagonal)
| Hạng | Nhãn Thực tế (True) | Nhãn Dự đoán Nhầm (Predicted) | Số mẫu nhầm | Nguyên nhân Hình học & Giải thích |
| :--- | :--- | :--- | :---: | :--- |
| 1 | `THREE_FINGER_POSE` | `FOUR_FINGER_POSE` | **2** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 2 | `TWO_FINGER_SCROLL_POSE` | `PINCH_MIDDLE_POSE` | **2** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 3 | `POINT_POSE` | `ZOOM_L_POSE` | **2** | Cả `POINT_POSE` `[0,1,0,0,0]` và `ZOOM_L_POSE` `[1,1,0,0,0]` đều duỗi ngón trỏ; chỉ khác nhau ở độ mở rộng của ngón cái theo trục ngang. |
| 4 | `FIST_IDLE_POSE` | `PINCH_INDEX_POSE` | **1** | Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp. |
| 5 | `FOUR_FINGER_POSE` | `THREE_FINGER_POSE` | **1** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 6 | `ZOOM_L_POSE` | `PINCH_INDEX_POSE` | **1** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 7 | `PINCH_MIDDLE_POSE` | `PINCH_INDEX_POSE` | **1** | Cả `PINCH_INDEX_POSE` và `PINCH_MIDDLE_POSE` đều thực hiện thao tác chụm ngón cái với một ngón dài ở vùng chuyển tiếp. |
| 8 | `PINCH_MIDDLE_POSE` | `POINT_POSE` | **1** | Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes). |