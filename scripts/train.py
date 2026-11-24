"""End-to-end training pipeline for customer churn prediction.

    python scripts/train.py                 # full run (~ a few minutes on a laptop CPU)
    python scripts/train.py --quick         # fast smoke run (small search, 3-fold CV)
    python scripts/train.py --download      # fetch the dataset into data/raw/ first

Pipeline: load -> validate -> clean -> feature engineering -> stratified split
-> imbalance experiment (CV) -> RandomizedSearchCV per model -> threshold
selection on out-of-fold predictions -> test evaluation -> feature importance
-> save best model.

The test set is used only for final reporting: model selection uses CV
ROC-AUC on the training set and the decision threshold is chosen on
out-of-fold training predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root (this file lives in scripts/)
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)

from src import data as data_mod
from src.evaluation import (
    METRIC_COLUMNS, best_f1_threshold, compute_metrics, model_feature_importance,
    permutation_importance_df, plot_confusion_matrix, plot_importance, plot_roc_pr,
    plot_threshold_tradeoff, threshold_table, to_markdown,
)
from src.features import add_features
from src.models import (
    IMBALANCE_STRATEGIES, PARAM_DISTRIBUTIONS, available_models, build_pipeline, supports_strategy,
)
from src.preprocessing import select_features

log = logging.getLogger("train")

PRETTY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}
CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc"}
# A non-default imbalance strategy is used for tuning only if it beats "none"
# on CV ROC-AUC by more than this margin.
STRATEGY_MIN_GAIN = 0.005


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=None, help="Path to the CSV (default: auto-detect in data/raw)")
    p.add_argument("--data-dir", type=Path, default=data_mod.RAW_DIR)
    p.add_argument("--download", action="store_true", help="Download the dataset if not present")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--models", nargs="+", default=None, help="Subset of models to train")
    p.add_argument("--n-iter", type=int, default=25, help="RandomizedSearchCV iterations per model")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--quick", action="store_true", help="Small, fast run for smoke testing")
    return p.parse_args(argv)


def load_dataset(args) -> pd.DataFrame:
    if args.data is None:
        try:
            data_mod.find_csv(args.data_dir)
        except data_mod.DataNotFoundError:
            if not args.download:
                raise
            data_mod.download_dataset(args.data_dir)
    raw = data_mod.load_raw(args.data, args.data_dir)
    report = data_mod.validate(raw)
    log.info("Validation report:\n%s", report.summary())
    if not report.ok:
        raise ValueError("Data validation failed; see report above.")
    df = data_mod.clean(raw)
    log.info("Clean data: %d rows. TotalCharges blanks filled with 0 for tenure=0: %d",
             len(df), report.blank_total_charges)
    return df, report


def imbalance_experiment(models, X, y, cv, pos_weight, args) -> pd.DataFrame:
    rows = []
    for name in models:
        for strategy in IMBALANCE_STRATEGIES:
            if not supports_strategy(name, strategy):
                continue
            pipe = build_pipeline(name, strategy, pos_weight=pos_weight, random_state=args.seed)
            t0 = time.time()
            res = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, n_jobs=args.n_jobs)
            row = {"model": PRETTY[name], "key": name, "strategy": strategy}
            for k in CV_SCORING:
                row[k] = res[f"test_{k}"].mean()
            row["roc_auc_std"] = res["test_roc_auc"].std()
            rows.append(row)
            log.info("  %-20s %-12s AUC=%.4f  recall=%.3f  precision=%.3f  (%.1fs)",
                     name, strategy, row["roc_auc"], row["recall"], row["precision"], time.time() - t0)
    return pd.DataFrame(rows)


def choose_strategy(imb: pd.DataFrame, name: str) -> str:
    sub = imb[imb.key == name].set_index("strategy")
    best = sub["roc_auc"].idxmax()
    if best != "none" and sub.loc[best, "roc_auc"] - sub.loc["none", "roc_auc"] > STRATEGY_MIN_GAIN:
        return best
    return "none"


def main(argv=None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.quick:
        args.n_iter, args.cv = min(args.n_iter, 4), 3

    fig_dir = args.reports_dir / "figures"
    for d in (args.models_dir, args.reports_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    df, vreport = load_dataset(args)
    df = add_features(df)
    X, y = select_features(df), df[data_mod.TARGET]

    churn_rate = float(y.mean())
    log.info("Class distribution: churn=%d (%.1f%%), no churn=%d", y.sum(), 100 * churn_rate, (1 - y).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    log.info("Train %d rows, test %d rows", len(X_train), len(X_test))
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    models = args.models or available_models()

    # ------------------------------------------------- imbalance experiment
    log.info("Imbalance experiment (%d-fold CV on train, default threshold 0.5)", args.cv)
    imb = imbalance_experiment(models, X_train, y_train, cv, pos_weight, args)
    imb.to_csv(args.reports_dir / "imbalance_cv.csv", index=False)

    # -------------------------------------------------------------- tuning
    tuned, rows_cv = {}, []
    for name in models:
        strategy = choose_strategy(imb, name)
        pipe = build_pipeline(name, strategy, pos_weight=pos_weight, random_state=args.seed)
        search = RandomizedSearchCV(
            pipe, PARAM_DISTRIBUTIONS[name], n_iter=args.n_iter, scoring="roc_auc",
            cv=cv, n_jobs=args.n_jobs, random_state=args.seed, refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # Out-of-fold probabilities on train -> F1-optimal decision threshold.
        oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba",
                                n_jobs=args.n_jobs)[:, 1]
        thr = best_f1_threshold(y_train, oof)
        tuned[name] = {"pipeline": best, "strategy": strategy, "threshold": thr, "oof": oof,
                       "test_proba": best.predict_proba(X_test)[:, 1]}
        rows_cv.append({
            "model": PRETTY[name], "key": name, "imbalance_strategy": strategy,
            "cv_roc_auc": search.best_score_,
            "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
            "oof_f1_threshold": thr,
            "best_params": {k.replace("clf__", ""): (v if isinstance(v, (str, int, type(None))) else float(v))
                            for k, v in search.best_params_.items()},
            "search_seconds": round(time.time() - t0, 1),
        })
        log.info("Tuned %-20s strategy=%-12s CV AUC=%.4f thr=%.3f (%.0fs) %s",
                 name, strategy, search.best_score_, thr, time.time() - t0, search.best_params_)

    cv_df = pd.DataFrame(rows_cv)
    cv_df.to_csv(args.reports_dir / "cv_results.csv", index=False)

    # ------------------------------------------------------------ selection
    # Model selection uses CV ROC-AUC on train only.
    best_key = cv_df.sort_values("cv_roc_auc", ascending=False).iloc[0]["key"]
    best = tuned[best_key]
    log.info("Selected model: %s (CV ROC-AUC %.4f)", best_key,
             cv_df.set_index("key").loc[best_key, "cv_roc_auc"])

    # ------------------------------------------------------- test evaluation
    rows_default, rows_tuned = [], []
    for name, t in tuned.items():
        rows_default.append({"Model": PRETTY[name], **compute_metrics(y_test, t["test_proba"], 0.5)})
        rows_tuned.append({"Model": PRETTY[name], "Threshold": t["threshold"],
                           **compute_metrics(y_test, t["test_proba"], t["threshold"])})
    test_default = pd.DataFrame(rows_default).set_index("Model")[METRIC_COLUMNS]
    test_tuned = pd.DataFrame(rows_tuned).set_index("Model")[["Threshold"] + METRIC_COLUMNS]
    test_default.to_csv(args.reports_dir / "test_metrics_threshold_0.5.csv")
    test_tuned.to_csv(args.reports_dir / "test_metrics_tuned_threshold.csv")

    imb_view = imb.drop(columns=["key"]).set_index("model")
    md = ["# Model comparison", "",
          f"Test set: {len(y_test)} customers, churn rate {y_test.mean():.3f}. "
          f"Selected model (by CV ROC-AUC on train): **{PRETTY[best_key]}**.", "",
          "## Test metrics at threshold 0.5", "", to_markdown(test_default), "",
          "## Test metrics at F1-optimal threshold (chosen on out-of-fold train predictions)", "",
          to_markdown(test_tuned), "",
          f"## Cross-validation (train, {args.cv}-fold, tuned)", "",
          to_markdown(cv_df.set_index("model")[["imbalance_strategy", "cv_roc_auc", "cv_roc_auc_std",
                                                 "oof_f1_threshold"]], ".4f"), "",
          f"## Imbalance experiment ({args.cv}-fold CV on train, untuned models, threshold 0.5)", "",
          to_markdown(imb_view, ".3f"), ""]
    (args.reports_dir / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # --------------------------------------------------------------- figures
    plot_roc_pr({PRETTY[k]: v["test_proba"] for k, v in tuned.items()}, y_test,
                path=fig_dir / "roc_pr_curves.png")
    _, cm_default = plot_confusion_matrix(y_test, best["test_proba"], 0.5,
                                          f"{PRETTY[best_key]} - threshold 0.50",
                                          fig_dir / "confusion_matrix_default.png")
    _, cm_tuned = plot_confusion_matrix(y_test, best["test_proba"], best["threshold"],
                                        f"{PRETTY[best_key]} - threshold {best['threshold']:.2f}",
                                        fig_dir / "confusion_matrix_tuned.png")
    plot_threshold_tradeoff(threshold_table(y_train, best["oof"]), best["threshold"],
                            f"Precision / recall trade-off - {PRETTY[best_key]} (out-of-fold, train)",
                            fig_dir / "threshold_tradeoff_oof.png")

    # ---------------------------------------------------- feature importance
    perm = permutation_importance_df(best["pipeline"], X_test, y_test, random_state=args.seed,
                                     n_repeats=5 if args.quick else 10)
    perm.to_csv(args.reports_dir / "permutation_importance.csv")
    plot_importance(perm["importance_mean"], f"Permutation importance - {PRETTY[best_key]} (test)",
                    "Mean drop in ROC-AUC", fig_dir / "permutation_importance.png",
                    err=perm["importance_std"])
    model_imps = {}
    for name, t in tuned.items():
        imp = model_feature_importance(t["pipeline"])
        if imp is None:
            continue
        model_imps[name] = imp
        imp.to_csv(args.reports_dir / f"model_importance_{name}.csv", header=["importance"])
        is_coef = name == "logistic_regression"
        plot_importance(imp.abs() if is_coef else imp,
                        f"{PRETTY[name]} - {'|coefficient|' if is_coef else 'built-in importance'}",
                        "|standardised coefficient|" if is_coef else "importance",
                        fig_dir / f"model_importance_{name}.png")

    # --------------------------------------------------------------- persist
    artifact = {
        "pipeline": best["pipeline"],
        "threshold": best["threshold"],
        "model_name": best_key,
        "features": list(X.columns),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, args.models_dir / "best_model.joblib")

    summary = {
        "dataset": {"rows": vreport.n_rows, "blank_total_charges_filled": vreport.blank_total_charges,
                    "duplicate_ids": vreport.duplicate_ids,
                    "duplicate_rows_excl_id_kept": vreport.duplicate_rows_excl_id,
                    "churn_rate": churn_rate, "train_rows": len(y_train), "test_rows": len(y_test)},
        "best_model": best_key,
        "best_params": cv_df.set_index("key").loc[best_key, "best_params"],
        "imbalance_strategy": best["strategy"],
        "cv_roc_auc": float(cv_df.set_index("key").loc[best_key, "cv_roc_auc"]),
        "threshold": best["threshold"],
        "test_metrics_threshold_0.5": test_default.loc[PRETTY[best_key]].to_dict(),
        "test_metrics_tuned_threshold": test_tuned.loc[PRETTY[best_key]].drop("Threshold").to_dict(),
        "confusion_matrix_threshold_0.5": cm_default.tolist(),
        "confusion_matrix_tuned_threshold": cm_tuned.tolist(),
        "top_permutation_features": perm["importance_mean"].head(10).round(4).to_dict(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float),
                                                   encoding="utf-8")

    pd.set_option("display.width", 140)
    print("\n=== Test metrics @ threshold 0.5 ===")
    print(test_default.round(4).to_string())
    print("\n=== Test metrics @ F1-optimal threshold (chosen on OOF train) ===")
    print(test_tuned.round(4).to_string())
    print(f"\nBest model: {PRETTY[best_key]}  threshold={best['threshold']:.3f}")
    print("Confusion matrix @0.5 [[TN, FP], [FN, TP]]:", cm_default.tolist())
    print(f"Confusion matrix @{best['threshold']:.2f}:", cm_tuned.tolist())
    print("\nTop permutation features (drop in ROC-AUC):")
    print(perm.head(10).round(4).to_string())
    return summary


if __name__ == "__main__":
    main()
