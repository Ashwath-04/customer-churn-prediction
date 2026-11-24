"""Metrics, threshold selection, plots and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

# Categorical palette in fixed slot order (colour follows the entity, never rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = {0: PALETTE[0], 1: PALETTE[1]}  # retained = blue, churned = orange
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def set_style():
    """Recessive axes/grid, thin marks, fixed categorical cycle."""
    from cycler import cycler
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TEXT, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "lines.linewidth": 2, "axes.prop_cycle": cycler(color=PALETTE),
        "legend.frameon": False, "font.size": 10,
    })


set_style()


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1. Call this on out-of-fold training
    predictions, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have one more element than thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, t)
        m["threshold"] = t
        m["flagged_pct"] = float((np.asarray(proba) >= t).mean())
        rows.append(m)
    return pd.DataFrame(rows).set_index("threshold")


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        vals = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- plots

def _save(fig, path: Path | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_confusion_matrix(y_true, proba, threshold: float = 0.5, title: str = "",
                          path: Path | None = None):
    cm = confusion_matrix(y_true, (np.asarray(proba) >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["No churn", "Churn"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(title or f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    _save(fig, path)
    return fig, cm


def plot_roc_pr(results: dict[str, np.ndarray], y_true, path: Path | None = None):
    """``results`` maps model name -> predicted probabilities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (name, proba) in enumerate(results.items()):
        c = PALETTE[i % len(PALETTE)]
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax1.plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
        p, r, _ = precision_recall_curve(y_true, proba)
        ax2.plot(r, p, color=c, label=name)
    ax1.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
    ax2.axhline(np.mean(y_true), color=MUTED, ls="--", lw=0.8, label="baseline (churn rate)")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_threshold_tradeoff(table: pd.DataFrame, chosen: float | None = None,
                            title: str = "Precision / recall trade-off",
                            path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, col in enumerate(("Precision", "Recall", "F1", "Accuracy")):
        ax.plot(table.index, table[col], color=PALETTE[i], marker="o", ms=4, label=col)
    if chosen is not None:
        ax.axvline(chosen, color=MUTED, ls="--", lw=1, label=f"chosen threshold={chosen:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title=title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_importance(series: pd.Series, title: str, xlabel: str, path: Path | None = None,
                    err: pd.Series | None = None, top: int = 15):
    s = series.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s.values, xerr=None if err is None else err.reindex(s.index).values,
            color=PALETTE[0], height=0.7, error_kw={"ecolor": MUTED, "lw": 1})
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------- importance

def permutation_importance_df(pipeline, X, y, scoring: str = "roc_auc", n_repeats: int = 10,
                              random_state: int = 42) -> pd.DataFrame:
    """Permutation importance on the *raw input columns* (each original
    feature is shuffled before preprocessing), so one-hot groups are scored
    together."""
    r = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
                               random_state=random_state, n_jobs=-1)
    return (pd.DataFrame({"importance_mean": r.importances_mean, "importance_std": r.importances_std},
                         index=X.columns)
            .sort_values("importance_mean", ascending=False))


def model_feature_importance(pipeline) -> pd.Series | None:
    """Impurity/gain importances (trees) or |coefficients| (logistic regression)
    on the transformed (one-hot) feature space."""
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    if hasattr(clf, "coef_"):
        return pd.Series(clf.coef_.ravel(), index=names).sort_values(key=np.abs, ascending=False)
    return None
