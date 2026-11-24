"""Column definitions and the scikit-learn preprocessing transformer.

The transformer is always fitted inside a Pipeline, so imputation statistics,
scaling parameters and one-hot categories are learned from training folds only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_spend", "service_count", "addon_count",
]

CATEGORICAL_FEATURES = [
    # demographics
    "gender", "SeniorCitizen", "Partner", "Dependents",
    # services
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    # account / billing
    "Contract", "PaperlessBilling", "PaymentMethod",
    # engineered
    "tenure_group",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        # drop="if_binary" keeps one column for Yes/No fields; unseen
        # categories at inference time are encoded as all zeros.
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[FEATURES]
