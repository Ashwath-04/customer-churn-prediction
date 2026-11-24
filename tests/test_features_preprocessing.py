import numpy as np

from src.data import clean
from src.features import add_features
from src.preprocessing import FEATURES, build_preprocessor, select_features


def test_engineered_features(raw_df):
    df = add_features(clean(raw_df))
    assert df["service_count"].between(0, 9).all()
    assert df["addon_count"].between(0, 6).all()
    assert (df.loc[df["InternetService"] == "No", "addon_count"] == 0).all()
    assert set(df["tenure_group"]) <= {"0-12", "13-24", "25-48", "49-72"}
    assert (df.loc[df["tenure"] == 0, "tenure_group"] == "0-12").all()
    assert (df.loc[df["tenure"] == 12, "tenure_group"] == "0-12").all()
    assert (df.loc[df["tenure"] == 13, "tenure_group"] == "13-24").all()
    zero = df["tenure"] == 0
    assert np.allclose(df.loc[zero, "avg_monthly_spend"], df.loc[zero, "MonthlyCharges"])
    assert np.isfinite(df["avg_monthly_spend"]).all()


def test_service_count_example():
    import pandas as pd
    row = pd.DataFrame([{
        "tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": 500.0,
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    }])
    out = add_features(row).iloc[0]
    assert out["addon_count"] == 2
    assert out["service_count"] == 5  # phone + multiple lines + internet + 2 add-ons
    assert out["avg_monthly_spend"] == 50.0


def test_preprocessor_no_nans_and_unknown_categories(raw_df):
    X = select_features(add_features(clean(raw_df)))
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    assert not np.isnan(Xt).any()

    X_new = X.head(3).copy()
    X_new.loc[X_new.index[0], "PaymentMethod"] = "Crypto"  # unseen category
    X_new.loc[X_new.index[1], "TotalCharges"] = np.nan       # missing value
    Xt_new = pre.transform(X_new)
    assert Xt_new.shape == (3, Xt.shape[1])
    assert not np.isnan(Xt_new).any()


def test_feature_list_has_no_target_or_id():
    assert "Churn" not in FEATURES and "customerID" not in FEATURES
