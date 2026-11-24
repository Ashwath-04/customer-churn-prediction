"""Data loading, validation and cleaning for the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Public copy of the dataset published by IBM on GitHub.
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ID_COL = "customerID"
TARGET = "Churn"

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

# Allowed values for categorical columns; anything else is flagged by validation.
ALLOWED_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


class DataNotFoundError(FileNotFoundError):
    pass


def find_csv(data_dir: Path | str = RAW_DIR) -> Path:
    """Auto-detect the Telco CSV inside ``data_dir``.

    Any ``*.csv`` whose header contains the expected columns is accepted, so the
    file can keep whatever name it was downloaded under (e.g. the Kaggle name
    ``WA_Fn-UseC_-Telco-Customer-Churn.csv``).
    """
    data_dir = Path(data_dir)
    candidates = sorted(data_dir.glob("*.csv"))
    # Prefer files whose name mentions churn/telco, then fall back to any CSV.
    candidates.sort(key=lambda p: 0 if any(k in p.name.lower() for k in ("churn", "telco")) else 1)
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:  # unreadable / not a CSV
            continue
        if set(EXPECTED_COLUMNS).issubset(header):
            return path
    raise DataNotFoundError(
        f"No Telco churn CSV found in {data_dir}.\n"
        "Download it from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)\n"
        f"or IBM ({DATASET_URL})\n"
        f"and place the .csv file inside {data_dir}, or run: python scripts/train.py --download"
    )


def download_dataset(data_dir: Path | str = RAW_DIR, url: str = DATASET_URL) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / DEFAULT_FILENAME
    logger.info("Downloading dataset from %s", url)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path: Path | str | None = None, data_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    path = Path(path) if path else find_csv(data_dir)
    logger.info("Loading %s", path)
    # Keep everything as read; TotalCharges is parsed explicitly in clean().
    return pd.read_csv(path)


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_values: dict[str, list] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    blank_total_charges: int = 0
    duplicate_ids: int = 0
    duplicate_rows_excl_id: int = 0
    negative_numeric: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_columns and not self.unexpected_values and not self.duplicate_ids

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows}, cols={self.n_cols}",
            f"missing columns: {self.missing_columns or 'none'}",
            f"unexpected categorical values: {self.unexpected_values or 'none'}",
            f"NaN counts (raw): {self.missing_values or 'none'}",
            f"blank TotalCharges strings: {self.blank_total_charges}",
            f"duplicate customerIDs: {self.duplicate_ids}",
            f"duplicate rows ignoring customerID: {self.duplicate_rows_excl_id}",
            f"negative numeric values: {self.negative_numeric or 'none'}",
        ]
        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationReport:
    """Schema / sanity checks on the raw frame. Does not modify ``df``."""
    report = ValidationReport(n_rows=len(df), n_cols=df.shape[1])
    report.missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    for col, allowed in ALLOWED_VALUES.items():
        if col in df.columns:
            bad = sorted(set(df[col].dropna().unique()) - allowed)
            if bad:
                report.unexpected_values[col] = bad

    na = df.isna().sum()
    report.missing_values = {c: int(n) for c, n in na.items() if n > 0}

    if "TotalCharges" in df.columns:
        tc = df["TotalCharges"]
        if not pd.api.types.is_numeric_dtype(tc):  # object or pandas>=3 "str" dtype
            report.blank_total_charges = int(tc.astype(str).str.strip().eq("").sum())

    if ID_COL in df.columns:
        report.duplicate_ids = int(df[ID_COL].duplicated().sum())
        report.duplicate_rows_excl_id = int(df.drop(columns=ID_COL).duplicated().sum())

    for col in ("tenure", "MonthlyCharges"):
        if col in df.columns:
            n = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            if n:
                report.negative_numeric[col] = n
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type fixes, missing-value handling, duplicate removal and binary target.

    * ``TotalCharges`` is stored as text with blank strings for customers whose
      tenure is 0 (no bill issued yet). Those are set to 0.0, which is the
      factually correct cumulative charge, rather than imputed with a statistic.
    * Exact duplicate customer IDs (none in the public file) are dropped.
      Rows that are identical apart from the ID are *kept* — they are different
      customers who happen to share the same attributes.
    * ``Churn`` Yes/No -> 1/0.
    """
    df = df.copy()
    tc = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", None), errors="coerce")
    zero_tenure = df["tenure"].eq(0)
    df["TotalCharges"] = tc.where(~(tc.isna() & zero_tenure), 0.0)
    # Any remaining NaNs (should not happen in the public file) are left for the
    # pipeline's SimpleImputer so the statistic is learned on training data only.

    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=ID_COL, keep="first")

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"].astype(str))
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df.reset_index(drop=True)


def load_clean(path: Path | str | None = None, data_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    raw = load_raw(path, data_dir)
    report = validate(raw)
    if not report.ok:
        raise ValueError(f"Data validation failed:\n{report.summary()}")
    return clean(raw)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    y = df[TARGET]
    return X, y
