# Telco Churn Project

Đề tài 5 — **Dự đoán và trực quan hóa tỷ lệ rời bỏ của khách hàng (Customer Churn) trong ngành viễn thông**.

## Dataset đã chốt
IBM Cognos Analytics Telco Customer Churn extended sample, gồm các bảng Demographics, Location, Population, Services và Status. Bộ dữ liệu cơ sở có 7.043 khách hàng và bản mở rộng cung cấp thông tin địa lý để phục vụ bản đồ.

## Mục tiêu
- Khám phá các yếu tố liên quan đến Customer Churn.
- Join tối thiểu 3 bảng qua `CustomerID`.
- Tạo `Tenure_Group`, `CLV_Estimated`, `Churn_Flag`.
- Xây dựng EDA tĩnh.
- Huấn luyện Logistic Regression.
- Dự đoán xác suất churn và phân nhóm Low / Medium / High Risk.
- Xây dựng Dashboard Streamlit + Plotly với 8 loại biểu đồ bắt buộc.

## Công nghệ
Python 3.11+, pandas, numpy, matplotlib, seaborn, scikit-learn, Plotly, Streamlit, openpyxl, joblib, Jupyter.

## Setup
```bash
python -m venv .venv
# Git Bash / macOS / Linux
source .venv/bin/activate
# Windows CMD/PowerShell: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data placement
Copy 5 file Excel vào:

```text
data/raw/
```

Tên file phải đúng theo `src/data_loader.py`.

## Run order
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

## Important modeling rule
Do not use outcome-derived or leakage-prone fields (`Churn Label`, `Churn Value`, `Churn Score`, `Churn Category`, `Churn Reason`, `Customer Status`) as model inputs. The baseline model also excludes `CLTV`.

## Current project status
**Phase 1–4 code framework is ready.** Actual execution depends on the five raw Excel files being present locally.

## Báo cáo và hình ảnh
`reports/report.docx` là living report. Sau mỗi notebook/milestone, cập nhật report cùng với output.

Các hình bằng chứng được chuẩn hóa trong `outputs/report_assets/` theo mã F00–F17. Xem `docs/report_figure_plan.md` để biết hình nào cần chụp, đặt ở mục nào và chụp vào thời điểm nào.

Quy tắc: không ghi số liệu minh họa thành kết quả chính thức; mọi metric/insight trong báo cáo phải lấy từ run thật trên 5 file raw.
