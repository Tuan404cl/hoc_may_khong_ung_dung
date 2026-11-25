# Phân tích Kết quả & Trường hợp Dự đoán sai - Decision Tree (CART, max_depth=12)

## 1. Chỉ số Đánh giá Tổng quát (Trên tập Kiểm thử 20%)
- **Accuracy:** `95.89%`
- **Precision (Macro):** `95.94%`
- **Recall (Macro):** `95.89%`
- **F1-Score (Macro):** `95.89%`
- **Tổng số mẫu kiểm thử:** `900` mẫu (`37` mẫu dự đoán sai)
- **Thời gian suy luận trung bình:** `0.0007 ms/frame`

## 2. Top các cặp Cử chỉ bị Nhầm lẫn nhiều nhất (Confusion Matrix Off-Diagonal)
| Hạng | Nhãn Thực tế (True) | Nhãn Dự đoán Nhầm (Predicted) | Số mẫu nhầm | Nguyên nhân Hình học & Giải thích |
| :--- | :--- | :--- | :---: | :--- |
| 1 | `OPEN_PALM_POSE` | `FOUR_FINGER_POSE` | **3** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 2 | `PINCH_INDEX_POSE` | `FIST_IDLE_POSE` | **3** | Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp. |
| 3 | `POINT_POSE` | `ZOOM_L_POSE` | **3** | Cả `POINT_POSE` `[0,1,0,0,0]` và `ZOOM_L_POSE` `[1,1,0,0,0]` đều duỗi ngón trỏ; chỉ khác nhau ở độ mở rộng của ngón cái theo trục ngang. |
| 4 | `POINT_POSE` | `PINCH_MIDDLE_POSE` | **3** | Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes). |
| 5 | `SHAKA_BRIGHTNESS_POSE` | `PINCH_MIDDLE_POSE` | **2** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 6 | `THREE_FINGER_POSE` | `FOUR_FINGER_POSE` | **2** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 7 | `TWO_FINGER_SCROLL_POSE` | `PINCH_INDEX_POSE` | **2** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 8 | `PINCH_MIDDLE_POSE` | `POINT_POSE` | **2** | Cả `POINT_POSE` và `PINCH_MIDDLE_POSE` đều giữ ngón trỏ duỗi thẳng `[0,1,0,0,0]`, khác biệt chủ yếu nằm ở khoảng cách chụm `dist_thumb_middle_norm` (dễ bị pha loãng dưới giả định độc lập của Naive Bayes). |