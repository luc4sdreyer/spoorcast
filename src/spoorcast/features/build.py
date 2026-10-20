"""The single entry point that turns canonical inputs into model features.

Training (:func:`spoorcast.data.dataset.build_dataset`) and serving
(:mod:`spoorcast.serving.predict`) both call :func:`build_features`. They differ only in where
the looked-up parts come from:

=============  ==========================================  ===================================
part           training path                               serving path
=============  ==========================================  ===================================
inputs         archive rows (:mod:`spoorcast.data.ingest`)  API request fields
weather        cached archive, :func:`lookup_weather`      a ``WeatherProvider``
aggregates     :func:`point_in_time_aggregates`            a ``FeatureLookup`` (snapshot)
=============  ==========================================  ===================================
"""

from __future__ import annotations

import pandas as pd

from spoorcast.features.aggregates import (
    AGGREGATE_FEATURES,
    DailyCounts,
    point_in_time_aggregates,
)
from spoorcast.features.calendar import CALENDAR_FEATURES, calendar_features
from spoorcast.features.weather import WEATHER_FEATURES, lookup_weather

# Canonical, request-level inputs. ``service_start_time`` may be missing (NaT).
INPUT_COLUMNS = [
    "station_code",
    "service_type",
    "company",
    "scheduled_departure",
    "service_start_time",
]

CATEGORICAL_FEATURES = ["station_code", "service_type", "company"]
TRIP_FEATURES = ["minutes_since_service_start"]
FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES,
    *CALENDAR_FEATURES,
    *TRIP_FEATURES,
    *WEATHER_FEATURES,
    *AGGREGATE_FEATURES,
]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def build_features(
    inputs: pd.DataFrame, weather: pd.DataFrame, aggregates: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the model feature frame, in :data:`FEATURE_COLUMNS` order.

    ``weather`` and ``aggregates`` must be aligned with ``inputs`` (same index) and hold the
    :data:`WEATHER_FEATURES` and :data:`AGGREGATE_FEATURES` columns respectively.
    """
    missing = [c for c in INPUT_COLUMNS if c not in inputs.columns]
    if missing:
        raise ValueError(f"build_features: missing input column(s) {missing}")
    for name, part in (("weather", weather), ("aggregates", aggregates)):
        if not part.index.equals(inputs.index):
            raise ValueError(f"build_features: {name} is not aligned with inputs")

    scheduled = inputs["scheduled_departure"]
    start = pd.to_datetime(inputs["service_start_time"], utc=True)
    minutes = (scheduled - start).dt.total_seconds() / 60.0

    features = pd.concat(
        [
            inputs[CATEGORICAL_FEATURES].astype(str),
            calendar_features(scheduled),
            pd.DataFrame({"minutes_since_service_start": minutes.astype("float64")}),
            weather[WEATHER_FEATURES].astype("float64"),
            aggregates[AGGREGATE_FEATURES].astype("float64"),
        ],
        axis=1,
    )
    return features[FEATURE_COLUMNS]


def training_features(
    departures: pd.DataFrame,
    weather_table: pd.DataFrame,
    daily: DailyCounts,
    window_days: int,
) -> pd.DataFrame:
    """Training path: features for archive departures (needs ``weather_point_id`` and
    ``service_date`` besides the canonical inputs)."""
    weather = lookup_weather(departures, weather_table)
    aggregates = point_in_time_aggregates(departures, daily, window_days)
    return build_features(departures[INPUT_COLUMNS], weather, aggregates)
