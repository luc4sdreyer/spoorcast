"""Validation of data on load. Every check fails loudly with a message saying what is wrong."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data does not look like what the pipeline expects."""


# Raw archive columns the pipeline reads, with the DuckDB type used to parse them.
REQUIRED_SERVICE_COLUMNS: dict[str, str] = {
    "Service:RDT-ID": "BIGINT",
    "Service:Date": "DATE",
    "Service:Type": "VARCHAR",
    "Service:Company": "VARCHAR",
    "Service:Train number": "BIGINT",
    "Stop:Station code": "VARCHAR",
    "Stop:Arrival time": "TIMESTAMPTZ",
    "Stop:Departure time": "TIMESTAMPTZ",
    "Stop:Departure delay": "BIGINT",
    "Stop:Departure cancelled": "BOOLEAN",
}
# Columns that only exist in some files. They are never used as features, so a missing one
# is fine; we only check that nothing *else* is missing.
OPTIONAL_SERVICE_COLUMNS = frozenset(
    {
        "Service:Completely cancelled",
        "Service:Partly cancelled",
        "Service:Maximum delay",
        "Stop:RDT-ID",
        "Stop:Station name",
        "Stop:Arrival delay",
        "Stop:Arrival cancelled",
        "Stop:Platform change",
        "Stop:Planned platform",
        "Stop:Actual platform",
    }
)

# Upper bounds on the share of nulls per column of the loaded departures.
DEPARTURE_MAX_NULL_RATE: dict[str, float] = {
    "service_id": 0.0,
    "service_date": 0.0,
    "service_type": 0.01,
    "company": 0.01,
    "station_code": 0.0,
    "scheduled_departure": 0.0,
    "service_start_time": 0.0,
    "departure_cancelled": 0.01,
    # The archive records a delay for (almost) every departure, cancelled ones included.
    "departure_delay_min": 0.05,
}

# Share of departures allowed outside [0, max_plausible_delay] before the file is rejected.
# The archive contains a handful of 24h-shifted records (delay ~1440) per month.
MAX_IMPLAUSIBLE_DELAY_FRACTION = 0.001


def check_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    present = set(columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise DataValidationError(
            f"{source}: missing expected column(s) {missing}. Found columns: {sorted(present)}"
        )


def check_null_rates(df: pd.DataFrame, max_null_rate: Mapping[str, float], source: str) -> None:
    if df.empty:
        raise DataValidationError(f"{source}: no rows")
    problems = []
    for column, limit in max_null_rate.items():
        rate = float(df[column].isna().mean())
        if rate > limit:
            problems.append(f"{column}: {rate:.2%} null (max {limit:.2%})")
    if problems:
        raise DataValidationError(f"{source}: null rates out of bounds: " + "; ".join(problems))


def validate_departures(df: pd.DataFrame, source: str, max_plausible_delay_min: int) -> None:
    """Validate departures loaded by :func:`spoorcast.data.ingest.read_departures`."""
    check_columns(df.columns, DEPARTURE_MAX_NULL_RATE, source)
    check_null_rates(df, DEPARTURE_MAX_NULL_RATE, source)

    for column in ("scheduled_departure", "service_start_time"):
        dtype = df[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype):
            raise DataValidationError(f"{source}: {column} must be tz-aware, got {dtype}")
    if not pd.api.types.is_numeric_dtype(df["departure_delay_min"]):
        raise DataValidationError(
            f"{source}: departure_delay_min must be numeric, got {df['departure_delay_min'].dtype}"
        )

    delay = df["departure_delay_min"].dropna()
    negative = int((delay < 0).sum())
    if negative:
        raise DataValidationError(f"{source}: {negative} departure(s) with a negative delay")
    implausible = float((delay > max_plausible_delay_min).mean()) if len(delay) else 0.0
    if implausible > MAX_IMPLAUSIBLE_DELAY_FRACTION:
        raise DataValidationError(
            f"{source}: {implausible:.3%} of departures have a delay above "
            f"{max_plausible_delay_min} min (max {MAX_IMPLAUSIBLE_DELAY_FRACTION:.3%})"
        )


def validate_stations(df: pd.DataFrame, source: str) -> None:
    check_columns(df.columns, ["code", "country", "geo_lat", "geo_lng"], source)
    check_null_rates(df, {"code": 0.0, "country": 0.0, "geo_lat": 0.0, "geo_lng": 0.0}, source)
    duplicated = df["code"][df["code"].duplicated()].tolist()
    if duplicated:
        raise DataValidationError(f"{source}: duplicate station codes {duplicated[:10]}")
    out_of_range = df[~df["geo_lat"].between(-90, 90) | ~df["geo_lng"].between(-180, 180)]
    if not out_of_range.empty:
        raise DataValidationError(
            f"{source}: invalid coordinates for {out_of_range['code'].tolist()[:10]}"
        )


# Physically plausible ranges for the weather variables (units as returned by Open-Meteo).
WEATHER_RANGES: dict[str, tuple[float, float]] = {
    "temperature_2m": (-40.0, 50.0),
    "precipitation": (0.0, 150.0),
    "snowfall": (0.0, 50.0),
    "wind_speed_10m": (0.0, 250.0),
    "wind_gusts_10m": (0.0, 300.0),
}


def validate_weather(df: pd.DataFrame, source: str) -> None:
    check_columns(df.columns, ["weather_point_id", "time_utc", *WEATHER_RANGES], source)
    check_null_rates(df, dict.fromkeys(WEATHER_RANGES, 0.05), source)
    for column, (low, high) in WEATHER_RANGES.items():
        values = df[column].dropna()
        bad = int((~values.between(low, high)).sum())
        if bad:
            raise DataValidationError(
                f"{source}: {bad} value(s) of {column} outside [{low}, {high}]"
            )
