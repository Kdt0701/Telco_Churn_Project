# Telecom Customer Churn — Predict & Visualize

Đề tài 5 — Lĩnh vực 1: Thương mại điện tử & Bán lẻ

## Mục tiêu
Phân tích các yếu tố liên quan đến Customer Churn, xây dựng Logistic Regression để dự đoán xác suất rời bỏ, phân nhóm rủi ro và xây dựng Dashboard tương tác.

## Dataset
Nguồn chính: IBM Cognos Analytics Telco customer churn sample.

Bộ dữ liệu mở rộng gồm 5 bảng:
- Demographics
- Location
- Population
- Services
- Status

Các file raw đặt trong `data/raw/`.

## Công nghệ
- Python 3.11+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- plotly
- streamlit
- openpyxl
- jupyter

## Pipeline
1. Load 5 bảng raw
2. Kiểm tra kích thước, kiểu dữ liệu, duplicate và missing values
3. Chuẩn hóa tên cột / khóa `Customer ID`
4. Join các bảng bằng khóa khách hàng
5. Feature Engineering
6. EDA tĩnh
7. Logistic Regression
8. Tạo `Churn_Probability` và `Risk_Level`
9. Xuất dữ liệu phục vụ Dashboard
10. Streamlit + Plotly Dashboard

## Cấu trúc
```text
telco_churn_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── dashboard/
├── docs/
├── reports/
├── outputs/
│   └── figures/
├── .gitignore
├── README.md
└── requirements.txt
```

## Lệnh cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Thứ tự chạy
```text
01_data_inspection.ipynb
        ↓
02_preprocessing.ipynb
        ↓
03_eda.ipynb
        ↓
04_logistic_regression.ipynb
        ↓
streamlit run dashboard/app.py
```

## Lưu ý về mô hình
Không dùng `Customer ID` làm biến đầu vào mô hình.
Không đưa các biến được tạo từ kết quả churn hoặc có nguy cơ rò rỉ mục tiêu vào Logistic Regression chỉ để làm điểm số đẹp.

## Mục tiêu đầu ra
`data/processed/final_dataset.csv`

Các trường dự kiến bổ sung:
- `Tenure_Group`
- `CLV_Estimated` (nếu cần tính từ dữ liệu dịch vụ)
- `Churn_Flag`
- `Churn_Probability`
- `Risk_Level`

## Dashboard bắt buộc
- Geographic Map
- Bar Chart
- Line Chart
- Donut Chart
- Heatmap
- Scatter Plot
- Treemap
- KPI / Gauge

Bộ lọc chính:
- Contract
- Payment Method
- Internet Service
- State / City
- Risk Level
