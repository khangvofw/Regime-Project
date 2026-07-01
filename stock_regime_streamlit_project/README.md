# Stock Regime Streamlit Dashboard

Dashboard Streamlit cho mô hình regime cổ phiếu ngân hàng và chứng khoán, dữ liệu full 2015–2026.

## Chạy local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy miễn phí trên Streamlit Community Cloud

1. Tạo GitHub repository mới.
2. Upload toàn bộ thư mục này lên repo.
3. Vào Streamlit Community Cloud → New app.
4. Chọn repo/branch và file `app.py`.
5. Bấm Deploy.

## Cấu trúc

```text
app.py
requirements.txt
.streamlit/config.toml
data/
  bank_panel.csv
  securities_panel.csv
```

## Nội dung app

- Tổng quan theo ngành.
- Chart regime đầy đủ từ 2015.
- Kiểm định xác định trạng thái hiện tại.
- Trend capture validation.
- Dự báo trạng thái xấu 20 phiên tới.
- Quy trình và hướng dẫn deploy.

