"""The same raw input must produce identical features on the training and serving paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import snapshot_as_of
from spoorcast.config import Settings
from spoorcast.data.dataset import read_partitions
from spoorcast.features.build import FEATURE_COLUMNS
from spoorcast.model import SpoorcastModel
from spoorcast.serving.app import create_app
from spoorcast.serving.predict import requests_to_inputs, serving_features
from spoorcast.serving.providers import (
    HistoricalWeatherProvider,
    LocalModelProvider,
    SnapshotFeatureLookup,
)
from spoorcast.serving.schemas import PredictionRequest

PARITY_DATE = pd.Timestamp("2026-02-02")


@pytest.fixture(scope="module")
def training_rows(built_settings: Settings) -> pd.DataFrame:
    dataset = read_partitions(built_settings.dataset_dir, ["2026-02"])
    rows = dataset[dataset["service_date"] == PARITY_DATE].reset_index(drop=True)
    assert len(rows) > 20
    return rows


@pytest.fixture(scope="module")
def lookup(built_settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Serving sees aggregates as of the snapshot date; use the rows' own service date.
    return snapshot_as_of(built_settings, PARITY_DATE.date(), tmp_path_factory.mktemp("snap"))


def _requests(rows: pd.DataFrame) -> list[PredictionRequest]:
    return [
        PredictionRequest(
            station_code=row.station_code,
            scheduled_departure=row.scheduled_departure.to_pydatetime(),
            service_type=row.service_type,
            company=row.company,
            service_start_time=row.service_start_time.to_pydatetime(),
        )
        for row in rows.itertuples()
    ]


def test_serving_features_equal_training_features(
    built_settings: Settings, training_rows: pd.DataFrame, lookup: Path
) -> None:
    feature_lookup = SnapshotFeatureLookup(lookup)
    feature_lookup.load()
    weather = HistoricalWeatherProvider(built_settings.weather_raw_dir)

    inputs = requests_to_inputs(_requests(training_rows), feature_lookup)
    served = serving_features(inputs, weather, feature_lookup)

    pd.testing.assert_frame_equal(
        served.reset_index(drop=True),
        training_rows[FEATURE_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )


def test_api_debug_features_equal_training_features(
    built_settings: Settings,
    training_rows: pd.DataFrame,
    lookup: Path,
    model_dir: Path,
) -> None:
    app = create_app(
        built_settings,
        model_provider=LocalModelProvider(model_dir),
        feature_lookup=SnapshotFeatureLookup(lookup),
        weather_provider=HistoricalWeatherProvider(built_settings.weather_raw_dir),
    )
    sample = training_rows.head(5)
    instances = [r.model_dump(mode="json") for r in _requests(sample)]
    with TestClient(app) as client:
        response = client.post("/v1/predictions/batch?debug=true", json={"instances": instances})
    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    served = pd.DataFrame([p["features_used"] for p in predictions])[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        served, sample[FEATURE_COLUMNS].reset_index(drop=True), check_dtype=False
    )

    # And the probability is the one the model gives for the training-path features.
    expected = SpoorcastModel.load(model_dir).predict_proba(sample[FEATURE_COLUMNS])
    served_proba = [p["probability_late"] for p in predictions]
    pd.testing.assert_series_equal(pd.Series(served_proba), pd.Series(expected))
