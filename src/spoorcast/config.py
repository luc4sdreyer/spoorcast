"""Configuration: environment-driven settings plus project-wide constants.

Every setting can be overridden with an environment variable prefixed ``SPOORCAST_``
(for example ``SPOORCAST_MODEL_DIR=/models/model``) or from a local ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Data sources -----------------------------------------------------------------------------

SERVICES_URL_TEMPLATES = (
    "https://opendata.rijdendetreinen.nl/public/services/services-{month}.csv.gz",
    "https://blobs.duckdb.org/nl-railway/services-{month}.csv.gz",  # mirror
)
STATIONS_URL = "https://opendata.rijdendetreinen.nl/public/stations/stations-2023-09.csv"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# The archive lags real time by a few days; months newer than this are not cached as final.
OPEN_METEO_ARCHIVE_LAG_DAYS = 7

LOCAL_TIMEZONE = "Europe/Amsterdam"


class Settings(BaseSettings):
    """Runtime settings shared by the CLI, the pipeline and the API."""

    model_config = SettingsConfigDict(
        env_prefix="SPOORCAST_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Data pipeline
    data_dir: Path = Path("data")
    late_threshold_min: int = 5
    max_plausible_delay_min: int = 720
    aggregate_window_days: int = 28

    # Serving
    model_dir: Path = Path("models/latest")
    snapshot_dir: Path | None = None  # defaults to <data_dir>/processed/feature_snapshot
    decision_threshold: float = 0.2
    weather_provider: Literal["open-meteo", "historical", "fixed"] = "open-meteo"
    inference_log_path: Path | None = None  # None means JSON lines on stdout
    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def services_raw_dir(self) -> Path:
        return self.raw_dir / "services"

    @property
    def stations_raw_path(self) -> Path:
        return self.raw_dir / "stations" / Path(STATIONS_URL).name

    @property
    def weather_raw_dir(self) -> Path:
        return self.raw_dir / "weather"

    @property
    def departures_dir(self) -> Path:
        """Cleaned, labelled departures (one partition per month), before feature building."""
        return self.processed_dir / "departures"

    @property
    def daily_stats_dir(self) -> Path:
        """Per-day counts used to compute the trailing aggregates."""
        return self.processed_dir / "daily_stats"

    @property
    def dataset_dir(self) -> Path:
        """Model-ready dataset: features + label, partitioned by ``year_month``."""
        return self.processed_dir / "dataset"

    @property
    def resolved_snapshot_dir(self) -> Path:
        return self.snapshot_dir or self.processed_dir / "feature_snapshot"
