"""Build the model-ready dataset: clean departures, join weather and aggregates, write Parquet.

Layout under ``data/processed``::

    departures/year_month=YYYY-MM/part-0.parquet    cleaned, labelled departures
    daily_stats/year_month=YYYY-MM/<aggregate>.parquet  per-day counts for the aggregates
    dataset/year_month=YYYY-MM/part-0.parquet       ids + features + label

Daily stats of every month already on disk are used for the trailing aggregates, so building
months incrementally ("a new month arrives") gives the same result as building them together.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from spoorcast.config import Settings
from spoorcast.data.ingest import clean_departures, load_stations, read_departures
from spoorcast.data.sources import services_path
from spoorcast.data.validate import DataValidationError, validate_weather
from spoorcast.data.weather import load_weather_table
from spoorcast.features.aggregates import (
    AGGREGATES,
    DailyCounts,
    combine_daily_counts,
    daily_counts,
)
from spoorcast.features.build import FEATURE_COLUMNS, training_features
from spoorcast.features.weather import WEATHER_FEATURES

logger = logging.getLogger(__name__)

ID_COLUMNS = [
    "service_id",
    "service_date",
    "train_number",
    "scheduled_departure",
    "service_start_time",
    "weather_point_id",
]
LABEL_COLUMNS = ["departure_delay_min", "late"]
DATASET_COLUMNS = [*ID_COLUMNS, *FEATURE_COLUMNS, *LABEL_COLUMNS]

# Fail the build when more departures than this lack weather (usually: not downloaded).
MAX_MISSING_WEATHER_FRACTION = 0.05


def _partition(base: Path, month: str) -> Path:
    return base / f"year_month={month}"


def _write_partition(df: pd.DataFrame, base: Path, month: str, name: str = "part-0") -> None:
    target = _partition(base, month)
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / f".{name}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.rename(target / f"{name}.parquet")


def read_partitions(base: Path, months: Sequence[str] | None = None) -> pd.DataFrame:
    """Read ``year_month`` partitions (all of them when ``months`` is None)."""
    available = list_months(base)
    wanted = available if months is None else [m for m in months if m in available]
    if not wanted:
        raise FileNotFoundError(f"No partitions for {months or 'any month'} under {base}")
    frames = []
    for month in wanted:
        frame = pd.read_parquet(_partition(base, month) / "part-0.parquet")
        frame["year_month"] = month
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def list_months(base: Path) -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in base.glob("year_month=*") if p.is_dir())


def load_daily_stats(settings: Settings) -> DailyCounts:
    """Daily counts of every month built so far."""
    months = list_months(settings.daily_stats_dir)
    if not months:
        raise FileNotFoundError(
            f"No daily stats under {settings.daily_stats_dir}; run `spoorcast build-dataset`"
        )
    parts = [
        {
            spec.name: pd.read_parquet(
                _partition(settings.daily_stats_dir, m) / f"{spec.name}.parquet"
            )
            for spec in AGGREGATES
        }
        for m in months
    ]
    return combine_daily_counts(parts)


def ingest_month(settings: Settings, month: str, stations: pd.DataFrame) -> pd.DataFrame:
    """Read, validate and clean one month; write its departures and daily stats."""
    path = services_path(settings, month)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `spoorcast download --months {month}`")
    departures, report = clean_departures(
        read_departures(path),
        stations,
        late_threshold_min=settings.late_threshold_min,
        max_plausible_delay_min=settings.max_plausible_delay_min,
        source=path.name,
    )
    logger.info(report.summary())
    departures["year_month"] = month
    _write_partition(departures.drop(columns="year_month"), settings.departures_dir, month)
    for name, counts in daily_counts(departures).items():
        _write_partition(counts, settings.daily_stats_dir, month, name=name)
    return departures


def build_dataset(months: Sequence[str], settings: Settings) -> Path:
    """Build dataset partitions for ``months``. Returns the dataset directory."""
    months = sorted(months)
    stations = load_stations(settings.stations_raw_path)
    weather = load_weather_table(settings.weather_raw_dir)
    validate_weather(weather, str(settings.weather_raw_dir))

    # Pass 1: clean every month and record daily stats (needed before any aggregate).
    for month in months:
        ingest_month(settings, month, stations)
    daily = load_daily_stats(settings)

    # Pass 2: features, one month at a time to bound memory.
    for month in months:
        departures = pd.read_parquet(_partition(settings.departures_dir, month) / "part-0.parquet")
        features = training_features(departures, weather, daily, settings.aggregate_window_days)
        missing_weather = float(features[WEATHER_FEATURES].isna().any(axis=1).mean())
        if missing_weather > MAX_MISSING_WEATHER_FRACTION:
            raise DataValidationError(
                f"{month}: {missing_weather:.1%} of departures have no weather; "
                f"run `spoorcast download --months {month}`"
            )
        dataset = pd.concat([departures[ID_COLUMNS], features, departures[LABEL_COLUMNS]], axis=1)[
            DATASET_COLUMNS
        ]
        partition = _partition(settings.dataset_dir, month)
        if partition.exists():
            shutil.rmtree(partition)
        _write_partition(dataset, settings.dataset_dir, month)
        logger.info(
            "Dataset %s: %d rows, late rate %.3f", month, len(dataset), dataset["late"].mean()
        )
    return settings.dataset_dir
