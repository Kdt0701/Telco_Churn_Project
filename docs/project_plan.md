# Project Plan — Đề tài 5 Customer Churn

## Phase 1 — Data inspection
1. Put five IBM Excel files into `data/raw/`.
2. Run `01_data_inspection.ipynb`.
3. Confirm row count, columns, missing values, duplicates and Join keys.
4. Confirm each customer table is one-to-one on `CustomerID`.

## Phase 2 — Preprocessing
1. Standardize text and `CustomerID`.
2. Convert numeric fields.
3. Convert blank `Total Charges` to numeric; zero-tenure blank values become 0, remaining missing values use the median.
4. Detect/cap charge outliers with IQR while retaining original fields.
5. Join Demographics + Location + Services + Status on `CustomerID`.
6. Join Population through `Zip Code` as contextual enrichment.
7. Create `Tenure_Group`, `CLV_Estimated`, `Churn_Flag`.
8. Export `data/processed/final_dataset.csv`.

## Phase 3 — EDA
Required static figures:
- Tenure distribution by churn with histogram/KDE.
- Churn rate by contract.
- Monthly charges boxplot by churn.
- Correlation heatmap.

## Phase 4 — Logistic Regression
- Target: `Churn_Flag`.
- Numeric preprocessing: median imputation + standardization.
- Categorical preprocessing: most-frequent imputation + one-hot encoding.
- Logistic Regression with reproducible seed and class balancing.
- Evaluate Accuracy, Precision, Recall, F1 and ROC-AUC.
- Score every customer with `Churn_Probability` and create `Risk_Level`.

## Phase 5 — Dashboard
Tool: Streamlit + Plotly.

Required visuals:
1. Geographic Map
2. Bar Chart
3. Line Chart
4. Donut Chart
5. Heatmap
6. Scatter Plot
7. Treemap
8. KPI / Gauge

Required interactions:
- Contract filter
- Payment Method filter
- Internet Service filter
- Risk filter
- State / City drill-down
- Plotly hover tooltip
- Cross-filtering through shared Streamlit filter state

## Phase 6 — Storytelling
Identify high-risk segments from the model and EDA. Phrase findings as associations/predictions, not causal proof. Convert findings into concrete retention actions.

## Phase 7 — IEEE report
Build the 7 required sections and include methodology, tables, figures, model results, limitations, demo link and usage instructions. Expand to at least 40 pages without padding unrelated content.
