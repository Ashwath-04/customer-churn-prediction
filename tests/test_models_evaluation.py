import json

import joblib
import numpy as np
import pytest

from src.data import clean
from src.evaluation import best_f1_threshold, compute_metrics, threshold_table
from src.features import add_features
from src.models import HAS_IMBLEARN, available_models, build_pipeline, supports_strategy
from src.preprocessing import select_features


@pytest.fixture
def xy(raw_df):
    df = add_features(clean(raw_df))
    return select_features(df), df["Churn"]


@pytest.mark.parametrize("name", available_models())
def test_every_model_fits_and_predicts(name, xy):
    X, y = xy
    pipe = build_pipeline(name).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()


def test_gradient_boosting_has_no_class_weight():
    assert not supports_strategy("gradient_boosting", "class_weight")
    with pytest.raises(ValueError):
        build_pipeline("gradient_boosting", "class_weight")


@pytest.mark.skipif(not HAS_IMBLEARN, reason="imbalanced-learn not installed")
def test_smote_only_resamples_during_fit(xy):
    X, y = xy
    pipe = build_pipeline("logistic_regression", "smote").fit(X, y)
    # predict path must return exactly one prediction per input row,
    # i.e. SMOTE is not applied at inference/validation time.
    assert len(pipe.predict(X)) == len(X)
    Xt, yt = pipe.named_steps["smote"].fit_resample(pipe.named_steps["pre"].transform(X), y)
    assert yt.mean() == pytest.approx(0.5)  # balanced only inside fit


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m["Recall"] == 1.0 and m["Precision"] == pytest.approx(2 / 3)
    assert m["ROC-AUC"] == 1.0
    assert best_f1_threshold(y, p) == pytest.approx(0.7)
    tbl = threshold_table(y, p, [0.5, 0.65])
    assert tbl.loc[0.65, "F1"] == 1.0


def test_train_script_quick_end_to_end(raw_csv_dir, tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train", script)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    summary = train.main([
        "--data-dir", str(raw_csv_dir), "--models-dir", str(tmp_path / "m"),
        "--reports-dir", str(tmp_path / "r"), "--quick", "--n-jobs", "1",
        "--models", "logistic_regression", "decision_tree",
    ])
    assert summary["best_model"] in {"logistic_regression", "decision_tree"}
    art = joblib.load(tmp_path / "m" / "best_model.joblib")
    assert 0 < art["threshold"] < 1
    assert json.loads((tmp_path / "r" / "metrics.json").read_text())["best_model"] == summary["best_model"]
    assert (tmp_path / "r" / "model_comparison.md").exists()
