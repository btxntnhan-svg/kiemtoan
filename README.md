# 🛡️ Hệ Thống Phát Hiện Giao Dịch Bất Thường (Transaction Anomaly Detection Web App)

Đây là ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit**, sử dụng mô hình học máy không giám sát **Isolation Forest** để phát hiện các giao dịch bất thường (nghi vấn gian lận) từ dữ liệu lịch sử.

Dự án được chuyển đổi và phát triển từ phân tích trong file Jupyter Notebook `phat_hien_bat_thuong.ipynb` sang một giao diện web ứng dụng hoàn chỉnh, trực quan và hiện đại.

---

## ✨ Tính Năng Chính

1. **Tải & Tiền xử lý dữ liệu động:**
   - Hỗ trợ người dùng tải lên tệp CSV giao dịch của riêng họ.
   - Có sẵn chế độ chạy thử dữ liệu demo mặc định (`transactions_Q1_demo.csv`).
   - Tự động trích xuất các đặc trưng như: **Giờ giao dịch**, **Giao dịch ngoài giờ** (định nghĩa là trước 6:00 sáng hoặc sau 18:00 chiều), và **Trạng thái nhân viên**.

2. **Huấn luyện Mô hình & Dự đoán Tốc độ cao:**
   - Sử dụng thuật toán **Isolation Forest** của thư viện `scikit-learn` huấn luyện cực nhanh trực tiếp trên web app.
   - Cho phép người dùng tinh chỉnh trực tiếp các siêu tham số mô hình trên thanh điều hướng bên trái: Tỷ lệ nhiễm bẩn (Contamination), Số lượng cây quyết định (n_estimators), và hạt giống ngẫu nhiên (Random State).
   - Tối ưu hóa hiệu năng phản hồi trang web bằng cách lưu trữ mô hình và dữ liệu đã huấn luyện trong **`st.session_state`**, tránh huấn luyện lại không cần thiết khi người dùng tương tác với giao diện.

3. **Phân loại Rủi ro theo 4 Mức độ:**
   - Dựa trên điểm số rủi ro (Anomaly Score) của nhóm giao dịch bất thường để chia làm 4 phân vị rủi ro rõ ràng:
     - **Khẩn cấp (Urgent):** Thuộc 25% nhóm có điểm dị biệt cao nhất (cực kỳ bất thường).
     - **Cao (High):** Phân vị từ 25% đến 50%.
     - **Trung bình (Medium):** Phân vị từ 50% đến 75%.
     - **Thấp (Low):** Phân vị từ 75% đến 100%.

4. **Trực quan hóa Dữ liệu Tương tác nâng cao:**
   - **Bản đồ 3D Không gian phân tách bất thường:** Vẽ biểu đồ không gian 3 chiều (Số tiền, Giờ giao dịch, Trạng thái nhân viên) bằng Plotly, tô màu rõ ràng theo trạng thái rủi ro của từng giao dịch.
   - Các biểu đồ thống kê 2D đa dạng: Số lượng giao dịch theo giờ (đánh dấu các giờ nghỉ ngoài hành chính), phân bổ bất thường theo kênh giao dịch, chi nhánh ngân hàng và phân phối số tiền giao dịch.

5. **Tra cứu & Xuất báo cáo:**
   - Bộ lọc chi tiết theo mức độ rủi ro, loại giao dịch, khoảng số tiền giao dịch và thanh tìm kiếm từ khóa giao dịch/khách hàng.
   - Cho phép xuất danh sách giao dịch bất thường đã lọc ra các định dạng **CSV** hoặc **Excel** để phục vụ công tác thanh tra/kiểm toán.

6. **Kiểm tra Giao dịch Đơn lẻ Thời gian thực:**
   - Form giả lập giao dịch (Số tiền, Giờ, Tài khoản nhân viên) để dự đoán và đưa ra kết quả phân tích rủi ro tức thì kèm khuyến nghị hành động cụ thể cho quản trị viên.

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ

### Yêu cầu hệ thống
- Máy tính cài đặt sẵn **Python 3.8** trở lên.

### Các bước cài đặt

1. **Tải mã nguồn về máy:**
   ```bash
   git clone <link-github-cua-ban>
   cd <thu-muc-du-an>
   ```

2. **Tạo môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   venv\Scripts\activate
   # Kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Ứng dụng sẽ tự động mở trên trình duyệt của bạn tại địa chỉ: `http://localhost:8501`.

---

## 📁 Cấu Trúc Dự Án

```text
├── .streamlit/
│   └── config.toml       # Cấu hình giao diện Dark Mode cao cấp cho Streamlit
├── app.py                # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt      # Danh sách thư viện Python cần cài đặt
├── README.md             # Tài liệu hướng dẫn sử dụng (file này)
├── transactions_Q1_demo.csv  # File dữ liệu giao dịch mẫu (49,996 dòng)
└── phat_hien_bat_thuong.ipynb # File Jupyter Notebook phân tích ban đầu
```

---

## 🧠 Tổng Quan về Thuật Toán & Đặc Trưng

Mô hình sử dụng thuật toán **Isolation Forest** là một phương pháp học máy không giám sát cực kỳ hiệu quả để phát hiện bất thường. Thuật toán hoạt động bằng cách cô lập (isolate) các điểm dữ liệu thông qua các cây quyết định ngẫu nhiên. Các điểm bất thường sẽ dễ dàng bị cô lập hơn và nằm gần gốc cây quyết định hơn, dẫn đến độ dài đường đi ngắn hơn và điểm anomaly score thấp hơn (âm).

Ba đặc trưng đầu vào chính bao gồm:
1. **Số tiền giao dịch (amount):** Các giao dịch có số tiền lớn vượt trội thường mang yếu tố rủi ro cao.
2. **Giờ giao dịch (gio_giao_dich):** Các giao dịch được thực hiện vào các khung giờ nghỉ hoặc đêm khuya (ngoại trừ giờ làm việc chuẩn 6:00 - 18:00) dễ bị phát hiện là bất thường.
3. **Trạng thái nhân viên (co nhan vien):** Xác định giao dịch có liên quan đến tài khoản của nhân viên nội bộ ngân hàng hay không.
