"""API tests with fake providers (no model file, no snapshot, no network)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from spoorcast.config import Settings
from spoorcast.features.aggregates import AGGREGATE_FEATURES
from spoorcast.features.build import FEATURE_COLUMNS
from spoorcast.serving.app import create_app
from spoorcast.serving.providers import (
    FeatureLookup,
    FixedWeatherProvider,
    InferenceLogger,
    LoadedModel,
    ModelProvider,
    NotReadyError,
    StationInfo,
)
from spoorcast.serving.schemas import MAX_BATCH_SIZE

REQUEST = {
    "station_code": "GD",
    "scheduled_departure": "2026-02-02T08:12:00+01:00",
    "service_type": "Sprinter",
    "company": "NS",
    "service_start_time": "2026-02-02T07:40:00+01:00",
}


class FakeModel:
    """Probability grows with the hour, so different inputs give different outputs."""

    metadata: ClassVar[dict[str, Any]] = {
        "model_version": "fake-1",
        "feature_columns": FEATURE_COLUMNS,
    }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert list(df.columns) == FEATURE_COLUMNS
        return (df["hour"].to_numpy(dtype=float) + 1) / 25


class FakeModelProvider:
    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def is_ready(self) -> bool:
        return self.loaded

    def get(self) -> LoadedModel:
        if not self.loaded:
            raise NotReadyError("not loaded")
        return LoadedModel(model=FakeModel(), version="fake-1")  # type: ignore[arg-type]


class FakeFeatureLookup:
    stations: ClassVar[dict[str, StationInfo]] = {
        "GD": StationInfo("GD", "Gouda", "rotterdam"),
        "WD": StationInfo("WD", "Woerden", "utrecht"),
    }

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def is_ready(self) -> bool:
        return self.loaded

    def station(self, station_code: str) -> StationInfo | None:
        return self.stations.get(station_code)

    def aggregates(self, inputs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: 0.05 for name in AGGREGATE_FEATURES}, index=inputs.index)

    def info(self) -> dict[str, Any]:
        return {"as_of": "2026-02-01"}


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, record: dict[str, Any]) -> None:
        self.records.append(record)


@pytest.fixture
def inference_logger() -> RecordingLogger:
    return RecordingLogger()


@pytest.fixture
def client(tmp_path: Any, inference_logger: RecordingLogger) -> Any:
    app = create_app(
        Settings(data_dir=tmp_path, decision_threshold=0.3),
        model_provider=FakeModelProvider(),
        feature_lookup=FakeFeatureLookup(),
        weather_provider=FixedWeatherProvider(),
        inference_logger=inference_logger,
    )
    with TestClient(app) as test_client:  # runs the lifespan, which loads the providers
        yield test_client


def test_fakes_implement_the_protocols() -> None:
    assert isinstance(FakeModelProvider(), ModelProvider)
    assert isinstance(FakeFeatureLookup(), FeatureLookup)
    assert isinstance(RecordingLogger(), InferenceLogger)


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {
        "ready": True,
        "model_loaded": True,
        "features_loaded": True,
    }


def test_ready_is_false_before_load(tmp_path: Any) -> None:
    app = create_app(
        Settings(data_dir=tmp_path),
        model_provider=FakeModelProvider(),
        feature_lookup=FakeFeatureLookup(),
        weather_provider=FixedWeatherProvider(),
        inference_logger=RecordingLogger(),
    )
    client = TestClient(app)  # not used as a context manager: lifespan (loading) never runs
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert client.post("/v1/predictions", json=REQUEST).status_code == 503


def test_ready_is_false_when_model_dir_is_missing(tmp_path: Any) -> None:
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "nope")
    app = create_app(settings, weather_provider=FixedWeatherProvider())
    with TestClient(app) as client:
        body = client.get("/ready").json()
    assert body == {"ready": False, "model_loaded": False, "features_loaded": False}


def test_predict_happy_path(client: TestClient, inference_logger: RecordingLogger) -> None:
    response = client.post("/v1/predictions", json=REQUEST)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["probability_late"] == pytest.approx(9 / 25)  # local hour 8
    assert body["late"] is True  # 0.36 >= threshold 0.3
    assert body["model_version"] == "fake-1"
    assert "features_used" not in body
    assert len(inference_logger.records) == 1
    record = inference_logger.records[0]
    assert record["prediction_id"] == body["prediction_id"]
    assert record["model_version"] == "fake-1"
    assert record["request"]["station_code"] == "GD"
    assert set(record["features"]) == set(FEATURE_COLUMNS)


def test_debug_returns_features(client: TestClient) -> None:
    body = client.post("/v1/predictions?debug=true", json=REQUEST).json()
    features = body["features_used"]
    assert features["hour"] == 8 and features["is_rush_hour"] == 1
    assert features["minutes_since_service_start"] == 32.0
    assert features["temperature_2m"] == FixedWeatherProvider.DEFAULTS["temperature_2m"]


def test_station_code_is_case_insensitive(client: TestClient) -> None:
    assert client.post("/v1/predictions", json={**REQUEST, "station_code": "gd"}).status_code == 200


def test_batch(client: TestClient, inference_logger: RecordingLogger) -> None:
    instances = [
        {**REQUEST, "scheduled_departure": f"2026-02-02T{hour:02d}:30:00+01:00"}
        for hour in range(8, 20)
    ]
    instances.append({**REQUEST, "station_code": "WD", "service_start_time": None})
    response = client.post("/v1/predictions/batch", json={"instances": instances})
    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert len(predictions) == len(instances)
    # Order is preserved: probability follows the hour.
    probabilities = [p["probability_late"] for p in predictions[:-1]]
    assert probabilities == sorted(probabilities)
    assert len(inference_logger.records) == len(instances)
    logged_ids = [r["prediction_id"] for r in inference_logger.records]
    assert logged_ids == [p["prediction_id"] for p in predictions]


def test_prediction_ids_are_unique(client: TestClient) -> None:
    instances = [REQUEST] * 50
    batch = client.post("/v1/predictions/batch", json={"instances": instances}).json()
    singles = [client.post("/v1/predictions", json=REQUEST).json() for _ in range(5)]
    ids = [p["prediction_id"] for p in batch["predictions"]] + [p["prediction_id"] for p in singles]
    assert len(set(ids)) == len(ids) == 55


def test_unknown_station_is_422(client: TestClient, inference_logger: RecordingLogger) -> None:
    response = client.post("/v1/predictions", json={**REQUEST, "station_code": "XYZ"})
    assert response.status_code == 422
    assert response.json()["detail"]["unknown_station_codes"] == ["XYZ"]

    batch = {"instances": [REQUEST, {**REQUEST, "station_code": "XYZ"}]}
    assert client.post("/v1/predictions/batch", json=batch).status_code == 422
    assert inference_logger.records == []


@pytest.mark.parametrize(
    "change",
    [
        {"scheduled_departure": "2026-02-02T08:12:00"},  # no offset
        {"scheduled_departure": "yesterday"},
        {"service_start_time": "2026-02-02T09:00:00+01:00"},  # after the departure
        {"station_code": ""},
    ],
)
def test_bad_input_is_422(client: TestClient, change: dict[str, str]) -> None:
    assert client.post("/v1/predictions", json={**REQUEST, **change}).status_code == 422


def test_batch_size_limits(client: TestClient) -> None:
    too_many = {"instances": [REQUEST] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/v1/predictions/batch", json=too_many).status_code == 422
    assert client.post("/v1/predictions/batch", json={"instances": []}).status_code == 422


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["model_version"] == "fake-1"
    assert body["decision_threshold"] == 0.3
    assert body["weather_provider"] == "fixed"
    assert body["feature_snapshot"] == {"as_of": "2026-02-01"}


def test_request_id_is_echoed(client: TestClient, inference_logger: RecordingLogger) -> None:
    response = client.post("/v1/predictions", json=REQUEST, headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert inference_logger.records[0]["request_id"] == "abc-123"
    assert client.get("/health").headers["X-Request-ID"]  # generated when absent


def test_openapi_has_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/predictions" in schema["paths"]
    assert schema["components"]["schemas"]["PredictionRequest"]["examples"]
