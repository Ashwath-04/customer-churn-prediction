"""Feature engineering.

All features here are deterministic, row-wise transformations of a single
customer's own record (no statistics are learned from the data), so applying
them before the train/test split cannot leak information.

Engineered features
-------------------
tenure_group
    Tenure bucketed into contract-length-like bands: 0-12, 13-24, 25-48, 49-72
    months. Churn is strongly non-linear in tenure (very high in year one,
    flattening afterwards); the bins let linear models capture that shape.
avg_monthly_spend
    TotalCharges / tenure, i.e. the customer's historical average bill. For
    tenure == 0 there is no history and the current MonthlyCharges is used.
    Justification: TotalCharges is (approximately) the sum of monthly bills, so
    the ratio is a well-defined average; it differs from MonthlyCharges when the
    customer's plan or price changed over time.
service_count
    Number of services subscribed out of 9: phone, multiple lines, internet,
    and the 6 internet add-ons (security, backup, device protection, tech
    support, streaming TV, streaming movies). Measures product engagement.
addon_count
    Number of the 6 internet add-on services subscribed (0 for customers
    without internet). Separates "bundle depth" from the base services.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]

ENGINEERED_FEATURES = ["tenure_group", "avg_monthly_spend", "service_count", "addon_count"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    tenure = df["tenure"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = df["TotalCharges"] / tenure
    df["avg_monthly_spend"] = np.where(tenure > 0, avg, df["MonthlyCharges"])

    addons = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    df["addon_count"] = addons.astype(int)
    df["service_count"] = (
        (df["PhoneService"] == "Yes").astype(int)
        + (df["MultipleLines"] == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + addons
    ).astype(int)
    return df
