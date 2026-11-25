# Phân tích Kết quả & Trường hợp Dự đoán sai - Logistic Regression (Multinomial L2)

## 1. Chỉ số Đánh giá Tổng quát (Trên tập Kiểm thử 20%)
- **Accuracy:** `97.89%`
- **Precision (Macro):** `97.93%`
- **Recall (Macro):** `97.89%`
- **F1-Score (Macro):** `97.89%`
- **Tổng số mẫu kiểm thử:** `900` mẫu (`19` mẫu dự đoán sai)
- **Thời gian suy luận trung bình:** `0.0006 ms/frame`

## 2. Top các cặp Cử chỉ bị Nhầm lẫn nhiều nhất (Confusion Matrix Off-Diagonal)
| Hạng | Nhãn Thực tế (True) | Nhãn Dự đoán Nhầm (Predicted) | Số mẫu nhầm | Nguyên nhân Hình học & Giải thích |
| :--- | :--- | :--- | :---: | :--- |
| 1 | `OPEN_PALM_POSE` | `FOUR_FINGER_POSE` | **3** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 2 | `THREE_FINGER_POSE` | `FOUR_FINGER_POSE` | **2** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 3 | `PINCH_INDEX_POSE` | `FIST_IDLE_POSE` | **2** | Khi chụm ngón cái và ngón trỏ nhưng các ngón còn lại gập sát, hình thái có nét gần với `FIST_IDLE_POSE` trong pha chuyển tiếp. |
| 4 | `FIST_IDLE_POSE` | `SHAKA_BRIGHTNESS_POSE` | **1** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 5 | `FIST_IDLE_POSE` | `PINCH_MIDDLE_POSE` | **1** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 6 | `OPEN_PALM_POSE` | `ZOOM_L_POSE` | **1** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |
| 7 | `FOUR_FINGER_POSE` | `THREE_FINGER_POSE` | **1** | Các cử chỉ nhiều ngón của Windows Touchscreen (`3 ngón`, `4 ngón`, `5 ngón`) có hình thái gần nhau khi ngón út hoặc ngón cái duỗi không hết. |
| 8 | `ZOOM_L_POSE` | `PINCH_INDEX_POSE` | **1** | Mẫu rơi vào giai đoạn chuyển tiếp tư thế tay (transitional pose) hoặc nhiễu độ sâu/góc xoay cổ tay lớn. |