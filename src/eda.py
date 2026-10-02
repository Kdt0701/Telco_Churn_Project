"""Static EDA plotting functions required by the assignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def churn_label(df):
    return df["Churn_Flag"].map({0: "Non-Churn", 1: "Churn"})


def plot_tenure_distribution(df, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Tenure in Months", hue=churn_label(df), kde=True, bins=25, stat="density", common_norm=False)
    plt.title("Tenure Distribution by Churn")
    plt.xlabel("Tenure (months)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_contract_churn(df, output_path: Path) -> None:
    summary = df.groupby("Contract", dropna=False)["Churn_Flag"].mean().sort_values(ascending=False).mul(100)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=summary.index, y=summary.values)
    plt.ylabel("Churn rate (%)")
    plt.xlabel("Contract")
    plt.title("Churn Rate by Contract")
    plt.xticks(rotation=15)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_monthly_charges_boxplot(df, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["Churn"] = churn_label(df)
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=plot_df, x="Churn", y="Monthly Charge")
    plt.title("Monthly Charges by Churn Status")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_correlation_heatmap(df, output_path: Path) -> None:
    numeric = df.select_dtypes(include="number").copy()
    preferred = [c for c in ["Age", "Number of Dependents", "Tenure in Months", "Avg Monthly GB Download", "Monthly Charge", "Total Charges", "Total Revenue", "Satisfaction Score", "CLTV", "Churn_Flag"] if c in numeric.columns]
    numeric = numeric[preferred] if preferred else numeric
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
