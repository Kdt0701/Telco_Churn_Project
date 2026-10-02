# Project Plan

## Phase 1 — Data Understanding
- Verify all five source files exist.
- Read workbook sheets and headers.
- Confirm row counts and unique customer IDs.
- Check join cardinality.

## Phase 2 — Preprocessing
- Standardize column names.
- Convert numeric fields.
- Handle missing values.
- Check outliers for Monthly Charges / Total Charges.
- Merge tables.
- Create Tenure_Group and other justified calculated fields.

## Phase 3 — EDA
- Tenure distribution by churn.
- Churn by contract.
- Monthly Charges by churn.
- Correlation heatmap.

## Phase 4 — Logistic Regression
- Train/test split.
- Encoding + scaling through sklearn Pipeline.
- Evaluate with confusion matrix, precision, recall, F1 and ROC-AUC.
- Produce probability for each customer.
- Create Low / Medium / High risk groups.

## Phase 5 — Dashboard
- KPI cards.
- Map.
- Bar, line, donut, heatmap, scatter, treemap.
- Multi-level filters.
- Cross-filter / drill-down behavior supported by the chosen dashboard design.

## Phase 6 — Storytelling
- Identify documented high-risk customer segments from the analysis.
- Connect findings to concrete retention actions.
- Clearly distinguish correlation/prediction from causal claims.

## Phase 7 — Report
- Build the 7 required sections and expand to at least 40 pages with methodology, figures, tables, implementation details and limitations.
