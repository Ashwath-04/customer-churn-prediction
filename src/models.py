"""Model zoo, class-imbalance strategies and small hyperparameter search spaces."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # optional dependency
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote")


def available_models() -> list[str]:
    names = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
    if HAS_XGB:
        names.append("xgboost")
    return names


def make_estimator(name: str, class_weight: bool = False, pos_weight: float = 1.0,
                   random_state: int = 42):
    """Return an untuned estimator with sensible defaults.

    ``class_weight=True`` re-weights the minority class. GradientBoosting has
    no class_weight parameter, so it raises ValueError for that strategy.
    """
    cw = "balanced" if class_weight else None
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=cw)
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight=cw, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=1,
                                      random_state=random_state)
    if name == "gradient_boosting":
        if class_weight:
            raise ValueError("GradientBoostingClassifier does not support class_weight")
        return GradientBoostingClassifier(random_state=random_state)
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=pos_weight if class_weight else 1.0,
            tree_method="hist", eval_metric="logloss", n_jobs=1,
            random_state=random_state,
        )
    raise KeyError(f"Unknown model: {name}")


def supports_strategy(name: str, strategy: str) -> bool:
    if strategy == "class_weight" and name == "gradient_boosting":
        return False
    if strategy == "smote" and not HAS_IMBLEARN:
        return False
    return True


def build_pipeline(name: str, strategy: str = "none", pos_weight: float = 1.0,
                   random_state: int = 42):
    """preprocess -> [SMOTE] -> classifier.

    SMOTE lives inside an imblearn Pipeline, whose samplers run only during
    ``fit``. Under cross-validation that means synthetic samples are generated
    from the training folds only; validation/test rows are never resampled.
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"strategy must be one of {IMBALANCE_STRATEGIES}")
    # Trees do not need scaling, but it is harmless and keeps one code path.
    pre = build_preprocessor(scale_numeric=True)
    clf = make_estimator(name, class_weight=(strategy == "class_weight"),
                         pos_weight=pos_weight, random_state=random_state)
    if strategy == "smote":
        if not HAS_IMBLEARN:
            raise ImportError("imbalanced-learn is required for SMOTE")
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=random_state)), ("clf", clf)])
    return Pipeline([("pre", pre), ("clf", clf)])


# Deliberately small search spaces over the parameters that matter most.
PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "clf__C": loguniform(1e-3, 1e2),
    },
    "decision_tree": {
        "clf__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
        "clf__min_samples_leaf": randint(1, 100),
        "clf__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [4, 6, 8, 10, 12, 16, None],
        "clf__min_samples_leaf": [1, 2, 4, 8, 16, 32],
        "clf__max_features": ["sqrt", 0.3, 0.5],
    },
    "gradient_boosting": {
        "clf__n_estimators": randint(100, 400),
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__min_samples_leaf": [1, 10, 30, 60],
    },
    "xgboost": {
        "clf__n_estimators": randint(100, 500),
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__max_depth": randint(2, 6),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__reg_lambda": loguniform(0.1, 10),
    },
}
