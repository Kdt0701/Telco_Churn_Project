"""Logistic Regression model training, evaluation and customer risk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn_Flag"

EXCLUDE = {
    "CustomerID",
    "Churn_Flag",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Category",
    "Churn Reason",
    "Customer Status",
    "CLTV",
    "Churn_Probability",
    "Risk_Level",
    "CLV_Estimated",
}

PREFERRED_FEATURES = [
    "Gender",
    "Age",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Number of Dependents",
    "State",
    "Internet Service",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Phone Service",
    "Multiple Lines",
    "Offer",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
]


def select_features(df: pd.DataFrame) -> list[str]:
    """Select model features that actually exist in the input dataframe."""
    features = [
        c for c in PREFERRED_FEATURES
        if c in df.columns and c not in EXCLUDE
    ]

    if not features:
        raise ValueError("No usable model features were found.")

    return features


def prepare_model_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pandas nullable dtypes before sending data to scikit-learn.

    In particular:
    - Numeric columns are converted to numeric and invalid values become NaN.
    - Categorical/string columns are converted to object.
    - pandas.NA is explicitly replaced with numpy.nan.
    """
    X = X.copy()

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # Keep numeric columns numeric.
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            # Convert nullable StringDtype / categorical columns
            # to ordinary Python objects.
            X[col] = X[col].astype(object)
            X[col] = X[col].where(X[col].notna(), np.nan)

    return X


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing + Logistic Regression pipeline."""

    # Ensure sklearn receives supported dtypes/missing-value representation.
    X = prepare_model_data(X)

    numeric = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=np.nan,
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "num",
                numeric_pipeline,
                numeric,
            )
        )

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=np.nan,
                        strategy="most_frequent",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=5,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame, dict]:
    """Train Logistic Regression, evaluate it and score all customers."""

    if TARGET not in df.columns:
        raise KeyError(f"Missing target column: {TARGET}")

    features = select_features(df)

    # Select features.
    X = df[features].copy()

    # Normalize nullable pandas values before train/test split.
    X = prepare_model_data(X)

    # Ensure target is a clean integer binary variable.
    y = pd.to_numeric(
        df[TARGET],
        errors="raise",
    ).astype(int)

    # Validate target.
    unique_targets = set(y.dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"{TARGET} must contain only 0/1 values. "
            f"Found: {sorted(unique_targets)}"
        )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X,
        y,
        df.index,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)

    # Train model.
    pipeline.fit(X_train, y_train)

    # Test predictions.
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            prob,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            pred,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            pred,
            zero_division=0,
        ),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": features,
    }

    # Score every customer.
    scored = df.copy()

    scored["Churn_Probability"] = pipeline.predict_proba(
        X
    )[:, 1]

    # Customer risk segmentation.
    scored["Risk_Level"] = pd.cut(
        scored["Churn_Probability"],
        bins=[
            -np.inf,
            0.30,
            0.60,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
        ],
        right=False,
    ).astype(str)

    # Record whether each row belongs to train/test set.
    scored["Model_Split"] = "Train"
    scored.loc[idx_test, "Model_Split"] = "Test"

    return pipeline, scored, metrics


def save_model(
    model: Pipeline,
    output_path: Path,
) -> None:
    """Save trained model to disk."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )