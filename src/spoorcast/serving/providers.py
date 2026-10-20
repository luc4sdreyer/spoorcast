"""Pluggable dependencies of the API. Each is a ``Protocol`` with a simple local implementation.

The API only depends on the protocols, so each local implementation can be swapped for a
platform-backed one without touching the endpoints (see ``docs/mlops-seams.md``).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, TextIO, runtime_checkable

import httpx
import pandas as pd

from spoorcast.config import OPEN_METEO_FORECAST_URL
from spoorcast.data.weather import (
    WEATHER_POINTS_BY_ID,
    WEATHER_VARIABLES,
    WeatherPoint,
    load_weather_table,
    parse_open_meteo,
)
from spoorcast.features.aggregates import lookup_snapshot
from spoorcast.features.snapshot import FeatureSnapshot, read_snapshot
from spoorcast.features.weather import WEATHER_FEATURES, lookup_weather
from spoorcast.model import SpoorcastModel

logger = logging.getLogger(__name__)


class NotReadyError(RuntimeError):
    """A dependency was used before it was loaded."""


# --- Model ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    model: SpoorcastModel
    version: str


@runtime_checkable
class ModelProvider(Protocol):
    def load(self) -> None: ...
    def is_ready(self) -> bool: ...
    def get(self) -> LoadedModel: ...


# MLOPS-SEAM: replace with a registry-backed ModelProvider (e.g. load the model behind a
# registry alias), or one that proxies predictions to a managed serving endpoint.
class LocalModelProvider:
    """Loads a ``SpoorcastModel`` saved with ``SpoorcastModel.save`` from a directory."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._loaded: LoadedModel | None = None

    def load(self) -> None:
        model = SpoorcastModel.load(self.model_dir)
        version = str(model.metadata.get("model_version", "unknown"))
        self._loaded = LoadedModel(model=model, version=version)
        logger.info("Model loaded", extra={"model_dir": str(self.model_dir), "version": version})

    def is_ready(self) -> bool:
        return self._loaded is not None

    def get(self) -> LoadedModel:
        if self._loaded is None:
            raise NotReadyError("Model is not loaded")
        return self._loaded


# --- Features ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StationInfo:
    station_code: str
    station_name: str
    weather_point_id: str


@runtime_checkable
class FeatureLookup(Protocol):
    def load(self) -> None: ...
    def is_ready(self) -> bool: ...
    def station(self, station_code: str) -> StationInfo | None: ...
    def aggregates(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Aggregate features aligned with ``inputs`` (canonical input columns)."""
        ...

    def info(self) -> dict[str, Any]: ...


# MLOPS-SEAM: replace with a FeatureLookup backed by an online feature table.
class SnapshotFeatureLookup:
    """Reads the Parquet snapshot written by ``spoorcast snapshot-features``."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self._snapshot: FeatureSnapshot | None = None
        self._stations: dict[str, StationInfo] = {}

    def load(self) -> None:
        snapshot = read_snapshot(self.snapshot_dir)
        self._stations = {
            row.station_code: StationInfo(row.station_code, row.station_name, row.weather_point_id)
            for row in snapshot.stations.itertuples(index=False)
        }
        self._snapshot = snapshot
        logger.info("Feature snapshot loaded", extra={"snapshot": snapshot.info})

    def is_ready(self) -> bool:
        return self._snapshot is not None

    def _require(self) -> FeatureSnapshot:
        if self._snapshot is None:
            raise NotReadyError("Feature snapshot is not loaded")
        return self._snapshot

    def station(self, station_code: str) -> StationInfo | None:
        self._require()
        return self._stations.get(station_code)

    def aggregates(self, inputs: pd.DataFrame) -> pd.DataFrame:
        return lookup_snapshot(inputs, self._require().aggregates)

    def info(self) -> dict[str, Any]:
        return dict(self._snapshot.info) if self._snapshot else {}


# --- Weather -------------------------------------------------------------------------------------


@runtime_checkable
class WeatherProvider(Protocol):
    name: str

    def weather(self, requests: pd.DataFrame) -> pd.DataFrame:
        """Weather features aligned with ``requests`` (``weather_point_id``,
        ``scheduled_departure``). Unknown values are NaN."""
        ...


class FixedWeatherProvider:
    """The same weather for every request. For tests and offline demos."""

    name = "fixed"
    DEFAULTS: ClassVar[dict[str, float]] = {
        "temperature_2m": 12.0,
        "precipitation": 0.0,
        "snowfall": 0.0,
        "wind_speed_10m": 15.0,
        "wind_gusts_10m": 30.0,
    }

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.values = {**self.DEFAULTS, **(values or {})}

    def weather(self, requests: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: [float(self.values[name])] * len(requests) for name in WEATHER_FEATURES},
            index=requests.index,
        )


class HistoricalWeatherProvider:
    """Observed weather from the cached Open-Meteo archive files (used for replays)."""

    name = "historical"

    def __init__(self, weather_dir: Path) -> None:
        self.weather_dir = Path(weather_dir)
        self._table: pd.DataFrame | None = None
        self._lock = threading.Lock()

    def _get_table(self) -> pd.DataFrame:
        with self._lock:
            if self._table is None:
                self._table = load_weather_table(self.weather_dir)
            return self._table

    def weather(self, requests: pd.DataFrame) -> pd.DataFrame:
        return lookup_weather(requests, self._get_table())


class OpenMeteoForecastProvider:
    """Hourly forecasts from the Open-Meteo forecast API, cached per weather point.

    One call covers the past two days and the coming week, so requests for departures in that
    range hit the cache. If the API is unreachable the weather features are left missing
    (the model handles that) rather than failing the prediction.
    """

    name = "open-meteo"

    def __init__(
        self,
        client: httpx.Client | None = None,
        ttl_seconds: float = 900.0,
        url: str = OPEN_METEO_FORECAST_URL,
    ) -> None:
        self.client = client or httpx.Client(timeout=10.0)
        self.ttl_seconds = ttl_seconds
        self.url = url
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def _fetch(self, points: list[WeatherPoint]) -> dict[str, pd.DataFrame]:
        response = self.client.get(
            self.url,
            params={
                "latitude": ",".join(str(p.lat) for p in points),
                "longitude": ",".join(str(p.lon) for p in points),
                "hourly": ",".join(WEATHER_VARIABLES),
                "timezone": "GMT",
                "wind_speed_unit": "kmh",
                "past_days": 2,
                "forecast_days": 7,
            },
        )
        response.raise_for_status()
        payload = response.json()
        payloads = payload if isinstance(payload, list) else [payload]
        return {
            p.point_id: parse_open_meteo(body, p.point_id)
            for p, body in zip(points, payloads, strict=True)
        }

    def _tables(self, point_ids: list[str]) -> list[pd.DataFrame]:
        now = time.monotonic()
        with self._lock:
            stale = [
                WEATHER_POINTS_BY_ID[p]
                for p in point_ids
                if p in WEATHER_POINTS_BY_ID
                and (p not in self._cache or now - self._cache[p][0] > self.ttl_seconds)
            ]
            if stale:
                try:
                    for point_id, table in self._fetch(stale).items():
                        self._cache[point_id] = (now, table)
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("Weather forecast unavailable: %s", exc)
            return [self._cache[p][1] for p in point_ids if p in self._cache]

    def weather(self, requests: pd.DataFrame) -> pd.DataFrame:
        tables = self._tables(sorted(requests["weather_point_id"].unique()))
        if not tables:
            return pd.DataFrame(
                {name: float("nan") for name in WEATHER_FEATURES}, index=requests.index
            )
        return lookup_weather(requests, pd.concat(tables, ignore_index=True))


# --- Inference logging ---------------------------------------------------------------------------


@runtime_checkable
class InferenceLogger(Protocol):
    def log(self, record: dict[str, Any]) -> None:
        """Record one prediction: request, features, prediction, ids, model version, time."""
        ...


# MLOPS-SEAM: replace with an InferenceLogger that writes to an inference table, so that
# predictions can be joined with outcomes (replay_labels.jsonl) for monitoring.
class JsonLinesInferenceLogger:
    """Writes one JSON object per prediction to a file (appending) or to stdout."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._stream: TextIO
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8")
        else:
            self._stream = sys.stdout

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self.path:
            self._stream.close()
