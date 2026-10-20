"""Training: time-based split, LightGBM model, baseline comparison.

``train`` *returns* everything worth tracking (model, metrics, params, window, input example)
and logs nothing anywhere, so experiment tracking can be wrapped around it.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from spoorcast import __version__
from spoorcast.data.dataset import list_months, read_dataset_info
from spoorcast.features.build import FEATURE_COLUMNS
from spoorcast.model import SpoorcastModel
from spoorcast.months import add_months
from spoorcast.training.baseline import StationHourBaseline
from spoorcast.training.evaluate import calibration_table, classification_metrics, prefixed

logger = logging.getLogger(__name__)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "cat_smooth": 20.0,
    "verbose": -1,
}


@dataclass
class TrainConfig:
    validate_month: str
    train_months: int = 12
    dataset_dir: Path = Path("data/processed/dataset")
    # Uniform sample of the training window, to fit in laptop memory. None: use every row.
    max_train_rows: int | None = 3_000_000
    seed: int = 42
    lgbm_params: dict[str, Any] = field(default_factory=dict)  # overrides the defaults

    def window(self) -> list[str]:
        return [add_months(self.validate_month, -n) for n in range(self.train_months, 0, -1)]


@dataclass
class TrainResult:
    model: SpoorcastModel
    baseline: StationHourBaseline
    metrics: dict[str, float]
    calibration: dict[str, pd.DataFrame]
    params: dict[str, Any]
    feature_columns: list[str]
    training_window: dict[str, Any]
    input_example: pd.DataFrame


def load_months(
    dataset_dir: Path,
    months: list[str],
    max_rows: int | None = None,
    seed: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read dataset months, optionally down-sampled uniformly to about ``max_rows`` rows."""
    columns = columns or [*FEATURE_COLUMNS, "late"]
    paths = [dataset_dir / f"year_month={m}" / "part-0.parquet" for m in months]
    total = sum(pq.read_metadata(p).num_rows for p in paths)
    fraction = 1.0 if not max_rows or total <= max_rows else max_rows / total
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        if fraction < 1.0:
            frame = frame.sample(frac=fraction, random_state=seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def resolve_window(config: TrainConfig) -> list[str]:
    available = set(list_months(config.dataset_dir))
    if config.validate_month not in available:
        raise ValueError(
            f"Validation month {config.validate_month} not in dataset {config.dataset_dir}"
        )
    wanted = config.window()
    months = [m for m in wanted if m in available]
    if not months:
        raise ValueError(f"No training months of {wanted[0]}..{wanted[-1]} in the dataset")
    if len(months) < len(wanted):
        logger.warning(
            "Training window %s..%s: %d of %d months available (missing: %s)",
            wanted[0],
            wanted[-1],
            len(months),
            len(wanted),
            ", ".join(sorted(set(wanted) - set(months))),
        )
    return months


def train(config: TrainConfig) -> TrainResult:
    """Train on months ``[t-N, t-1]`` and validate on month ``t`` (no random splits)."""
    train_months = resolve_window(config)
    dataset_info = read_dataset_info(config.dataset_dir)
    params = {**DEFAULT_LGBM_PARAMS, "random_state": config.seed, **config.lgbm_params}

    train_df = load_months(config.dataset_dir, train_months, config.max_train_rows, config.seed)
    valid_df = load_months(config.dataset_dir, [config.validate_month])
    logger.info(
        "Training on %d rows (%s), validating on %d", len(train_df), train_months, len(valid_df)
    )

    created_at = datetime.now(UTC)
    sha = git_sha()
    training_window = {"train_months": train_months, "validate_month": config.validate_month}
    model = SpoorcastModel.fit(train_df[FEATURE_COLUMNS], train_df["late"], params)
    baseline = StationHourBaseline().fit(train_df, train_df["late"])

    y_valid = valid_df["late"].to_numpy()
    model_proba = model.predict_proba(valid_df[FEATURE_COLUMNS])
    baseline_proba = baseline.predict_proba(valid_df)
    metrics = {
        **prefixed(classification_metrics(y_valid, model_proba), "model"),
        **prefixed(classification_metrics(y_valid, baseline_proba), "baseline"),
        "n_train": float(len(train_df)),
    }
    calibration = {
        "model": calibration_table(y_valid, model_proba),
        "baseline": calibration_table(y_valid, baseline_proba),
    }
    model.update_metadata(
        {
            "model_version": f"{created_at:%Y%m%dT%H%M%SZ}-{sha or 'nogit'}",
            "created_at": created_at.isoformat(),
            "package_version": __version__,
            "git_sha": sha,
            "late_threshold_min": dataset_info["late_threshold_min"],
            "aggregate_window_days": dataset_info["aggregate_window_days"],
            "training_window": training_window,
            "metrics": metrics,
            "params": params,
        }
    )
    return TrainResult(
        model=model,
        baseline=baseline,
        metrics=metrics,
        calibration=calibration,
        params=params,
        feature_columns=list(FEATURE_COLUMNS),
        training_window=training_window,
        input_example=valid_df[FEATURE_COLUMNS].head(5).reset_index(drop=True),
    )


def evaluate_saved_model(
    model_dir: Path, month: str, dataset_dir: Path, max_train_rows: int | None = 3_000_000
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    """Score a saved model and the baseline (refit on the model's training window) on a month."""
    model = SpoorcastModel.load(model_dir)
    train_months = model.metadata.get("training_window", {}).get("train_months", [])
    df = load_months(dataset_dir, [month])
    y = df["late"].to_numpy()
    model_proba = model.predict_proba(df[model.feature_columns])
    metrics = prefixed(classification_metrics(y, model_proba), "model")
    calibration = {"model": calibration_table(y, model_proba)}
    available = [m for m in train_months if m in set(list_months(dataset_dir))]
    if available:
        train_df = load_months(dataset_dir, available, max_train_rows)
        baseline = StationHourBaseline().fit(train_df, train_df["late"])
        baseline_proba = baseline.predict_proba(df)
        metrics |= prefixed(classification_metrics(y, baseline_proba), "baseline")
        calibration["baseline"] = calibration_table(y, baseline_proba)
    return metrics, calibration
