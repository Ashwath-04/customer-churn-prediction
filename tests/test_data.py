import pandas as pd
import pytest

from src import data


def test_find_csv_autodetects_any_name(raw_csv_dir):
    (raw_csv_dir / "notes.csv").write_text("a,b\n1,2\n")
    assert data.find_csv(raw_csv_dir).name == "telco_customer_churn.csv"


def test_find_csv_missing_gives_instructions(tmp_path):
    with pytest.raises(data.DataNotFoundError, match="data"):
        data.find_csv(tmp_path)


def test_validate_flags_blanks_and_bad_values(raw_df):
    report = data.validate(raw_df)
    assert report.ok
    assert report.blank_total_charges == (raw_df["tenure"] == 0).sum()

    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Weekly"
    bad = pd.concat([bad, bad.iloc[[1]]])
    report = data.validate(bad)
    assert not report.ok
    assert report.unexpected_values == {"Contract": ["Weekly"]}
    assert report.duplicate_ids == 1


def test_clean_types_and_target(raw_df):
    df = data.clean(raw_df)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all()
    assert set(df["Churn"].unique()) <= {0, 1}
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_clean_drops_duplicate_ids(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5]])
    assert len(data.clean(dup)) == len(raw_df)


def test_real_dataset_if_present():
    try:
        path = data.find_csv()
    except data.DataNotFoundError:
        pytest.skip("raw dataset not downloaded")
    df = data.load_clean(path)
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0
    assert 0.25 < df["Churn"].mean() < 0.28
