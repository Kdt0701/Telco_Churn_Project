"""Load and inspect the five IBM Telco Customer Churn source tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FILES: Dict[str, str] = {
    "demographics": "Telco_customer_churn_demographics.xlsx",
    "location": "Telco_customer_churn_location.xlsx",
    "population": "Telco_customer_churn_population.xlsx",
    "services": "Telco_customer_churn_services.xlsx",
    "status": "Telco_customer_churn_status.xlsx",
}


def load_table(name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one named source table from Excel."""
    if name not in RAW_FILES:
        raise KeyError(f"Unknown table '{name}'. Expected: {sorted(RAW_FILES)}")
    path = raw_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw file: {path}\n"
            "Download the IBM Telco churn source files and place them in data/raw/."
        )
    return pd.read_excel(path)


def load_all_tables(raw_dir: Path = RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Load all five tables."""
    return {name: load_table(name, raw_dir) for name in RAW_FILES}


def inspect_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a compact table-level inspection report."""
    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": len(df.columns),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tables = load_all_tables()
    print(inspect_tables(tables).to_string(index=False))
