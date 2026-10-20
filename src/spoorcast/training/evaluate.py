"""Evaluation metrics for probabilistic late-departure predictions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

_EPS = 1e-6


def classification_metrics(y_true: np.ndarray | pd.Series, proba: np.ndarray) -> dict[str, float]:
    """ROC AUC, PR AUC (average precision), Brier score and log loss.

    Ranking metrics are NaN when ``y_true`` has a single class (as on tiny samples).
    """
    y = np.asarray(y_true, dtype="int64")
    p = np.clip(np.asarray(proba, dtype="float64"), _EPS, 1 - _EPS)
    both_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else math.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "base_rate": float(y.mean()) if len(y) else math.nan,
        "n": float(len(y)),
    }


def calibration_table(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Observed late rate per decile of predicted probability."""
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype="int64"), "p": np.asarray(proba)})
    # Rank first so that ties (common for the baseline) do not collapse the quantile edges.
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=bins, labels=False) + 1
    table = frame.groupby("bin").agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        min_predicted=("p", "min"),
        max_predicted=("p", "max"),
    )
    return table.reset_index()


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}
