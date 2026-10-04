"""Cleaning, joining and feature engineering for IBM Telco churn tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import load_all_tables
from .features import add_feature_engineering, normalize_customer_id

DROP_META = ["Count", "Quarter"]

# Missing values that are meaningful business states in this dataset.
STRUCTURAL_MISSING_LABELS = {
    "Offer": "No Offer",
    "Internet Type": "No Internet",
    "Churn Category": "No Churn",
    "Churn Reason": "No Churn Reason",
}


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text and convert pandas nullable strings to object + np.nan.

    Keeping ordinary object dtype here avoids pandas.NA leaking into
    scikit-learn transformers while still preserving missing values for
    explicit imputation later in the modeling pipeline.
    """
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.isna(), np.nan).astype(object)
    return df


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill dataset-specific structural blanks with explicit labels."""
    df = df.copy()
    for col, label in STRUCTURAL_MISSING_LABELS.items():
        if col in df.columns:
            df[col] = df[col].replace({"": np.nan, " ": np.nan})
            df[col] = df[col].fillna(label)
    return df


def clean_services(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_customer_id(clean_text(df))
    drop_cols = [c for c in DROP_META if c in df.columns]
    df = df.drop(columns=drop_cols)

    numeric_cols = [
        "Tenure in Months",
        "Avg Monthly Long Distance Charges",
        "Avg Monthly GB Download",
        "Number of Referrals",
        "Monthly Charge",
        "Total Charges",
        "Total Refunds",
        "Total Extra Data Charges",
        "Total Long Distance Charges",
        "Total Revenue",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace({"": np.nan, " ": np.nan}),
                errors="coerce",
            )

    # Blank Total Charges is meaningful for zero-tenure customers.
    if "Total Charges" in df.columns:
        tenure = pd.to_numeric(
            df.get("Tenure in Months", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        zero_tenure = tenure.fillna(0).eq(0)
        df.loc[zero_tenure & df["Total Charges"].isna(), "Total Charges"] = 0

        # Any remaining missing Total Charges are imputed with the median.
        if df["Total Charges"].isna().any():
            df["Total Charges"] = df["Total Charges"].fillna(
                df["Total Charges"].median()
            )

    return fill_structural_missing(df)


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_customer_id(clean_text(df))
    df = df.drop(columns=[c for c in DROP_META if c in df.columns])

    for col in ["Age", "Number of Dependents"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_customer_id(clean_text(df))
    for col in ["Zip Code", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=[c for c in DROP_META if c in df.columns])


def clean_status(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_customer_id(clean_text(df))
    return fill_structural_missing(
        df.drop(columns=[c for c in DROP_META if c in df.columns])
    )


def clean_population(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_text(df).copy()
    if "Zip Code" not in df.columns:
        raise KeyError("Population table must contain Zip Code.")

    df["Zip Code"] = pd.to_numeric(df["Zip Code"], errors="coerce")
    if "Population" in df.columns:
        df["Population"] = pd.to_numeric(df["Population"], errors="coerce")

    # Population is ZIP-level. Aggregate defensively before joining to
    # customer rows so the merge remains many-to-one.
    value_cols = [c for c in ["Population"] if c in df.columns]
    df = (
        df[["Zip Code", *value_cols]]
        .dropna(subset=["Zip Code"])
        .groupby("Zip Code", as_index=False)
        .mean(numeric_only=True)
    )
    return df


def _assert_unique_key(df: pd.DataFrame, key: str, table_name: str) -> None:
    if key not in df.columns:
        raise KeyError(f"{table_name} must contain {key}.")

    dup = df[key].duplicated().sum()
    if dup:
        raise ValueError(
            f"{table_name} has {dup} duplicate {key} values; "
            "fix cardinality before Join."
        )


def cap_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cap extreme numeric values using IQR fences while preserving originals."""
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            df[f"{col}_Capped"] = s
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_Capped"] = s.clip(lower=lower, upper=upper)

    return df


def build_customer_churn_dataset(raw_dir: Path) -> pd.DataFrame:
    tables = load_all_tables(raw_dir)

    demo = clean_demographics(tables["demographics"])
    loc = clean_location(tables["location"])
    services = clean_services(tables["services"])
    status = clean_status(tables["status"])
    population = clean_population(tables["population"])

    for name, table in [
        ("demographics", demo),
        ("location", loc),
        ("services", services),
        ("status", status),
    ]:
        _assert_unique_key(table, "CustomerID", name)

    customer = demo.merge(
        loc,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    customer = customer.merge(
        services,
        on="CustomerID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_service"),
    )

    customer = customer.merge(
        status,
        on="CustomerID",
        how="left",
        validate="one_to_one",
        suffixes=("", "_status"),
    )

    if "Zip Code" in customer.columns and not population.empty:
        customer = customer.merge(
            population,
            on="Zip Code",
            how="left",
            validate="many_to_one",
        )

    customer = add_feature_engineering(customer)
    customer = cap_iqr(customer, ["Monthly Charge", "Total Charges"])

    # Keep one row per customer and remove exact duplicate customer rows defensively.
    customer = (
        customer.drop_duplicates(subset=["CustomerID"])
        .reset_index(drop=True)
    )

    # Final dataset-level structural missing cleanup for known categories.
    customer = fill_structural_missing(customer)

    return customer


def save_processed_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "processed" / "final_dataset.csv"

    df = build_customer_churn_dataset(project_root / "data" / "raw")
    save_processed_dataset(df, output)

    print(f"Saved {len(df):,} customers to {output}")
