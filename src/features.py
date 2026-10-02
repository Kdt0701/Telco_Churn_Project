"""Feature engineering for the customer-level Telco churn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common Customer ID spellings to CustomerID."""
    df = df.copy()
    candidates = ["Customer ID", "CustomerID", "customer_id", "customerid"]
    found = next((c for c in candidates if c in df.columns), None)
    if found is None:
        raise KeyError(f"Customer ID column not found. Columns: {list(df.columns)}")
    if found != "CustomerID":
        df = df.rename(columns={found: "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip()
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create transparent calculated fields required by the assignment."""
    df = df.copy()

    if "Tenure in Months" in df.columns:
        tenure = pd.to_numeric(df["Tenure in Months"], errors="coerce")
    elif "Tenure Months" in df.columns:
        tenure = pd.to_numeric(df["Tenure Months"], errors="coerce")
    else:
        raise KeyError("A tenure column is required for Tenure_Group.")

    if "Tenure in Months" not in df.columns:
        df["Tenure in Months"] = tenure

    df["Tenure_Group"] = pd.cut(
        tenure,
        bins=[-np.inf, 11, 35, np.inf],
        labels=["<1 year", "1-3 years", ">3 years"],
    ).astype("string")

    if "Monthly Charge" in df.columns:
        monthly = pd.to_numeric(df["Monthly Charge"], errors="coerce")
    else:
        monthly = pd.Series(np.nan, index=df.index)

    df["CLV_Estimated"] = (monthly.fillna(0) * tenure.fillna(0)).round(2)

    if "Churn Value" in df.columns:
        df["Churn_Flag"] = pd.to_numeric(df["Churn Value"], errors="coerce").fillna(0).astype(int)
    elif "Churn Label" in df.columns:
        df["Churn_Flag"] = (
            df["Churn Label"].astype("string").str.strip().str.lower().eq("yes").astype(int)
        )
    else:
        raise KeyError("Churn Value or Churn Label is required for Churn_Flag.")

    return df
