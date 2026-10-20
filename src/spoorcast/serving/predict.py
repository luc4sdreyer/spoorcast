"""Serving path: API requests -> canonical inputs -> features -> predictions -> inference log."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from spoorcast.features.build import INPUT_COLUMNS, build_features
from spoorcast.serving.providers import FeatureLookup, InferenceLogger, LoadedModel, WeatherProvider
from spoorcast.serving.schemas import PredictionRequest, PredictionResponse


class UnknownStationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = sorted(set(codes))
        super().__init__(f"Unknown station code(s): {', '.join(self.codes)}")


def requests_to_inputs(
    requests: Sequence[PredictionRequest], feature_lookup: FeatureLookup
) -> pd.DataFrame:
    """Canonical inputs (plus ``weather_point_id``) for a list of API requests."""
    stations = {r.station_code: feature_lookup.station(r.station_code) for r in requests}
    unknown = [code for code, info in stations.items() if info is None]
    if unknown:
        raise UnknownStationError(unknown)
    return pd.DataFrame(
        {
            "station_code": [r.station_code for r in requests],
            "service_type": [r.service_type for r in requests],
            "company": [r.company for r in requests],
            "scheduled_departure": pd.to_datetime(
                [r.scheduled_departure for r in requests], utc=True
            ),
            "service_start_time": pd.to_datetime(
                [r.service_start_time for r in requests], utc=True
            ),
            "weather_point_id": [
                stations[r.station_code].weather_point_id  # type: ignore[union-attr]
                for r in requests
            ],
        }
    )


def serving_features(
    inputs: pd.DataFrame, weather_provider: WeatherProvider, feature_lookup: FeatureLookup
) -> pd.DataFrame:
    weather = weather_provider.weather(inputs)
    aggregates = feature_lookup.aggregates(inputs)
    return build_features(inputs[INPUT_COLUMNS], weather, aggregates)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, float | np.floating) and np.isnan(value):
            out[key] = None
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def predict(
    requests: Sequence[PredictionRequest],
    *,
    model: LoadedModel,
    feature_lookup: FeatureLookup,
    weather_provider: WeatherProvider,
    inference_logger: InferenceLogger,
    decision_threshold: float,
    request_id: str | None = None,
    include_features: bool = False,
) -> list[PredictionResponse]:
    inputs = requests_to_inputs(requests, feature_lookup)
    features = serving_features(inputs, weather_provider, feature_lookup)
    probabilities = model.model.predict_proba(features)
    timestamp = datetime.now(UTC).isoformat()

    responses = []
    for request, feature_row, probability in zip(
        requests, features.to_dict(orient="records"), probabilities, strict=True
    ):
        features_used = _jsonable(feature_row)
        response = PredictionResponse(
            prediction_id=uuid.uuid4(),
            probability_late=float(probability),
            late=bool(probability >= decision_threshold),
            model_version=model.version,
            features_used=features_used if include_features else None,
        )
        inference_logger.log(
            {
                "prediction_id": str(response.prediction_id),
                "request_id": request_id,
                "timestamp": timestamp,
                "model_version": model.version,
                "request": request.model_dump(mode="json"),
                "features": features_used,
                "probability_late": response.probability_late,
                "late": response.late,
                "decision_threshold": decision_threshold,
            }
        )
        responses.append(response)
    return responses
