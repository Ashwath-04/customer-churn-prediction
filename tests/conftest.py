import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ALLOWED_VALUES  # noqa: E402


def make_synthetic_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Random rows with the raw Telco schema (TotalCharges as text, blanks for tenure 0)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"customerID": [f"{i:04d}-TEST" for i in range(n)]})
    for col, allowed in ALLOWED_VALUES.items():
        df[col] = rng.choice(sorted(allowed), size=n)
    # keep service columns internally consistent
    no_inet = df["InternetService"] == "No"
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]:
        df.loc[no_inet, col] = "No internet service"
        df.loc[~no_inet & (df[col] == "No internet service"), col] = "No"
    no_phone = df["PhoneService"] == "No"
    df.loc[no_phone, "MultipleLines"] = "No phone service"
    df.loc[~no_phone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"

    df["SeniorCitizen"] = rng.integers(0, 2, size=n)
    df["tenure"] = rng.integers(0, 73, size=n)
    df["MonthlyCharges"] = rng.uniform(18, 120, size=n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df["TotalCharges"] = total.astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    # make churn learnable: short tenure + month-to-month -> churn
    logit = -0.5 - 0.05 * df["tenure"] + 1.5 * (df["Contract"] == "Month-to-month")
    df["Churn"] = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture
def raw_df():
    return make_synthetic_raw()


@pytest.fixture
def raw_csv_dir(tmp_path, raw_df):
    raw_df.to_csv(tmp_path / "telco_customer_churn.csv", index=False)
    return tmp_path
