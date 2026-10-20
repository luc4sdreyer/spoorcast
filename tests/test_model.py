from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spoorcast.config import Settings
from spoorcast.data.dataset import read_partitions
from spoorcast.features.build import FEATURE_COLUMNS
from spoorcast.model import METADATA_FILE, ModelInputError, SpoorcastModel
from spoorcast.training.train import TrainConfig, TrainResult


@pytest.fixture(scope="module")
def features(built_settings: Settings) -> pd.DataFrame:
    return read_partitions(built_settings.dataset_dir)[FEATURE_COLUMNS]


def test_round_trip_gives_identical_predictions(
    train_result: TrainResult, saved_model: SpoorcastModel, features: pd.DataFrame
) -> None:
    np.testing.assert_array_equal(
        train_result.model.predict_proba(features), saved_model.predict_proba(features)
    )
    assert saved_model.feature_columns == FEATURE_COLUMNS
    assert saved_model.input_schema == train_result.model.input_schema


def test_metadata_file(model_dir: Path, train_result: TrainResult) -> None:
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["late_threshold_min"] == 5
    assert metadata["training_window"] == {
        "train_months": ["2026-01"],
        "validate_month": "2026-02",
    }
    assert metadata["metrics"]["model_pr_auc"] == pytest.approx(
        train_result.metrics["model_pr_auc"]
    )
    for key in ("package_version", "model_version", "git_sha", "created_at", "params"):
        assert key in metadata


def test_column_order_does_not_matter(saved_model: SpoorcastModel, features: pd.DataFrame) -> None:
    shuffled = features[list(reversed(FEATURE_COLUMNS))]
    np.testing.assert_array_equal(
        saved_model.predict_proba(features), saved_model.predict_proba(shuffled)
    )


def test_unseen_category_still_predicts(
    saved_model: SpoorcastModel, features: pd.DataFrame
) -> None:
    row = features.head(1).copy()
    row["station_code"] = "NEWSTATION"
    proba = saved_model.predict_proba(row)
    assert proba.shape == (1,) and 0.0 <= proba[0] <= 1.0


def test_missing_column_is_rejected(saved_model: SpoorcastModel, features: pd.DataFrame) -> None:
    with pytest.raises(ModelInputError, match="temperature_2m"):
        saved_model.predict_proba(features.drop(columns="temperature_2m"))


def test_train_result_contents(train_result: TrainResult) -> None:
    assert train_result.feature_columns == FEATURE_COLUMNS
    assert list(train_result.input_example.columns) == FEATURE_COLUMNS
    for prefix in ("model", "baseline"):
        for metric in ("roc_auc", "pr_auc", "brier", "log_loss"):
            assert not math.isnan(train_result.metrics[f"{prefix}_{metric}"])
    assert set(train_result.calibration) == {"model", "baseline"}
    assert train_result.calibration["model"]["n"].sum() == train_result.metrics["model_n"]


def test_window_is_time_based() -> None:
    config = TrainConfig(validate_month="2026-08", train_months=3)
    assert config.window() == ["2026-05", "2026-06", "2026-07"]
