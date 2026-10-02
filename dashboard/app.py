"""Interactive Streamlit + Plotly dashboard for Telco Customer Churn."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "model_scored.csv"

st.set_page_config(page_title="Telecom Customer Churn", layout="wide")
st.title("Telecom Customer Churn Dashboard")
st.caption("Đề tài 5 — Dự đoán và trực quan hóa tỷ lệ rời bỏ khách hàng")

if not DATA_PATH.exists():
    st.error("Chưa có data/processed/model_scored.csv. Hãy chạy notebook 02 rồi 04 trước.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Sidebar multi-level filters
st.sidebar.header("Bộ lọc")
for col, label in [("Contract", "Contract"), ("Payment Method", "Payment Method"), ("Internet Service", "Internet Service"), ("Risk_Level", "Risk Level"), ("State", "State")]:
    if col in df.columns:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, options, default=options)
        df = df[df[col].astype(str).isin(selected)]

if "State" in df.columns and "City" in df.columns:
    states = sorted(df["State"].dropna().astype(str).unique())
    drill_state = st.selectbox("Drill-down: chọn State", ["All"] + states)
    if drill_state != "All":
        cities = sorted(df.loc[df["State"].astype(str) == drill_state, "City"].dropna().astype(str).unique())
        drill_city = st.selectbox("Drill-down: chọn City", ["All"] + cities)
        df = df[df["State"].astype(str) == drill_state]
        if drill_city != "All":
            df = df[df["City"].astype(str) == drill_city]

if df.empty:
    st.warning("Không còn dữ liệu sau khi áp dụng bộ lọc.")
    st.stop()

# KPI
churn_rate = df["Churn_Flag"].mean() if "Churn_Flag" in df.columns else 0
risk_revenue = df.loc[df["Risk_Level"].eq("High"), "Monthly Charge"].sum() if "Monthly Charge" in df.columns else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Churn Rate", f"{churn_rate:.1%}")
k3.metric("High Risk", f"{(df['Risk_Level'] == 'High').sum():,}")
k4.metric("Monthly Revenue at High Risk", f"${risk_revenue:,.0f}")

# 1 Map
if {"Latitude", "Longitude"}.issubset(df.columns):
    map_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if not map_df.empty:
        fig = px.scatter_map(
            map_df, lat="Latitude", lon="Longitude", color="Risk_Level" if "Risk_Level" in map_df else None,
            size="Monthly Charge" if "Monthly Charge" in map_df else None,
            hover_name="CustomerID", hover_data=[c for c in ["State", "City", "Contract", "Internet Service", "Churn_Probability"] if c in map_df.columns],
            zoom=3.5, height=520, title="Customer Churn Risk — Geographic Map"
        )
        fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    # 2 Bar
    if {"Internet Service", "Churn_Flag"}.issubset(df.columns):
        bar = df.groupby("Internet Service", dropna=False)["Churn_Flag"].mean().reset_index()
        bar["Churn Rate"] = bar["Churn_Flag"] * 100
        fig = px.bar(bar, x="Internet Service", y="Churn Rate", text="Churn Rate", title="Churn Rate by Internet Service")
        st.plotly_chart(fig, use_container_width=True)
with c2:
    # 3 Line
    line = df.groupby("Tenure in Months", dropna=False)["Churn_Flag"].mean().reset_index()
    line["Churn Rate"] = line["Churn_Flag"] * 100
    fig = px.line(line, x="Tenure in Months", y="Churn Rate", markers=True, title="Churn Trend by Tenure")
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    # 4 Donut
    donut = df["Churn_Flag"].map({0: "Non-Churn", 1: "Churn"}).value_counts().reset_index()
    donut.columns = ["Status", "Customers"]
    fig = px.pie(donut, names="Status", values="Customers", hole=0.55, title="Overall Churn vs Non-Churn")
    st.plotly_chart(fig, use_container_width=True)
with c4:
    # 5 Heatmap-like pivot
    if {"Contract", "Payment Method", "Churn_Flag"}.issubset(df.columns):
        heat = df.pivot_table(index="Contract", columns="Payment Method", values="Churn_Flag", aggfunc="mean") * 100
        fig = px.imshow(heat, text_auto=".1f", aspect="auto", title="Churn Rate: Contract × Payment Method", labels=dict(color="Churn %"))
        st.plotly_chart(fig, use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    # 6 Scatter
    fig = px.scatter(df, x="Tenure in Months", y="Total Charges", color="Churn_Flag", hover_data=[c for c in ["CustomerID", "Contract", "Internet Type", "Risk_Level"] if c in df.columns], title="Tenure vs Total Charges")
    st.plotly_chart(fig, use_container_width=True)
with c6:
    # 7 Treemap
    tree_parts = [c for c in ["Premium Tech Support", "Online Security", "Online Backup"] if c in df.columns]
    if tree_parts and "Monthly Charge" in df.columns:
        tm = df.copy()
        tm["Lost_Revenue_Risk"] = tm["Monthly Charge"] * tm["Churn_Probability"]
        fig = px.treemap(tm, path=tree_parts, values="Lost_Revenue_Risk", title="Estimated Revenue at Churn Risk by Service")
        st.plotly_chart(fig, use_container_width=True)

# 8 KPI / Gauge
threshold = 0.90
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_rate * 100,
    title={"text": "Current Churn Rate (%)"},
    gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"width": 4}, "value": threshold * 100}},
))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top high-risk customers")
cols = [c for c in ["CustomerID", "State", "City", "Contract", "Payment Method", "Internet Type", "Tenure in Months", "Monthly Charge", "Churn_Probability", "Risk_Level"] if c in df.columns]
if "Churn_Probability" in df.columns:
    st.dataframe(df.sort_values("Churn_Probability", ascending=False)[cols].head(20), use_container_width=True)
