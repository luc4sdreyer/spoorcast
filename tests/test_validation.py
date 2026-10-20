"""Bad input data must fail loudly, with a message that says what is wrong."""

from __future__ import annotations

import gzip
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from conftest import FIXTURES
from spoorcast.config import Settings
from spoorcast.data.dataset import build_dataset
from spoorcast.data.ingest import clean_departures, load_stations, read_departures
from spoorcast.data.validate import DataValidationError, validate_weather
from spoorcast.data.weather import load_weather_table

SERVICES = FIXTURES / "services" / "services-2026-01.csv.gz"


def rewrite_csv(src: Path, dest: Path, edit: Callable[[pd.DataFrame], None]) -> Path:
    with gzip.open(src, "rt") as fh:
        frame = pd.read_csv(fh, dtype=str, keep_default_na=False)
    edit(frame)
    with gzip.open(dest, "wt") as fh:
        frame.to_csv(fh, index=False)
    return dest


@pytest.fixture
def departures() -> pd.DataFrame:
    return read_departures(SERVICES)


@pytest.fixture
def stations() -> pd.DataFrame:
    return load_stations(FIXTURES / "stations" / "stations-2023-09.csv")


def clean(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    out, _ = clean_departures(df, stations, late_threshold_min=5, max_plausible_delay_min=720)
    return out


def test_missing_required_column_is_reported(tmp_path: Path) -> None:
    path = rewrite_csv(
        SERVICES,
        tmp_path / "services.csv.gz",
        lambda f: f.drop(columns="Stop:Departure delay", inplace=True),
    )
    with pytest.raises(DataValidationError, match=r"missing expected column.*Departure delay"):
        read_departures(path)


def test_missing_optional_platform_columns_are_fine(tmp_path: Path) -> None:
    def drop_platforms(frame: pd.DataFrame) -> None:
        frame.drop(
            columns=["Stop:Platform change", "Stop:Planned platform", "Stop:Actual platform"],
            inplace=True,
        )

    path = rewrite_csv(SERVICES, tmp_path / "services.csv.gz", drop_platforms)
    assert len(read_departures(path)) == len(read_departures(SERVICES))


def test_unparseable_values_are_reported(tmp_path: Path) -> None:
    def corrupt(frame: pd.DataFrame) -> None:
        frame.loc[3, "Stop:Departure delay"] = "five"

    path = rewrite_csv(SERVICES, tmp_path / "services.csv.gz", corrupt)
    with pytest.raises(DataValidationError, match="could not parse"):
        read_departures(path)


def test_negative_delay_fails(departures: pd.DataFrame, stations: pd.DataFrame) -> None:
    departures.loc[0, "departure_delay_min"] = -3
    with pytest.raises(DataValidationError, match="negative delay"):
        clean(departures, stations)


def test_too_many_implausible_delays_fail(departures: pd.DataFrame, stations: pd.DataFrame) -> None:
    departures.loc[:20, "departure_delay_min"] = 5000
    with pytest.raises(DataValidationError, match="delay above 720"):
        clean(departures, stations)


def test_null_rate_out_of_bounds_fails(departures: pd.DataFrame, stations: pd.DataFrame) -> None:
    departures["station_code"] = departures["station_code"].astype(object)
    departures.loc[:5, "station_code"] = None
    with pytest.raises(DataValidationError, match="station_code"):
        clean(departures, stations)


def test_naive_timestamps_fail(departures: pd.DataFrame, stations: pd.DataFrame) -> None:
    departures["scheduled_departure"] = departures["scheduled_departure"].dt.tz_localize(None)
    with pytest.raises(DataValidationError, match="tz-aware"):
        clean(departures, stations)


def test_exclusions_and_label(departures: pd.DataFrame, stations: pd.DataFrame) -> None:
    out, report = clean_departures(
        departures, stations, late_threshold_min=5, max_plausible_delay_min=720
    )
    assert report.dropped["cancelled"] == int(departures["departure_cancelled"].sum())
    assert report.rows_out == len(out)
    assert not out["service_type"].str.contains("bus", case=False).any()
    assert set(out["station_code"]) <= set(
        stations.loc[stations["country"] == "NL", "station_code"]
    )
    assert (out["late"] == (out["departure_delay_min"] >= 5)).all()


def test_duplicate_station_codes_fail(tmp_path: Path) -> None:
    lines = (FIXTURES / "stations" / "stations-2023-09.csv").read_text().splitlines()
    path = tmp_path / "stations.csv"
    path.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(DataValidationError, match="duplicate station codes"):
        load_stations(path)


def test_implausible_weather_fails() -> None:
    weather = load_weather_table(FIXTURES / "weather")
    weather.loc[0, "temperature_2m"] = 90.0
    with pytest.raises(DataValidationError, match="temperature_2m outside"):
        validate_weather(weather, "test")


def test_build_without_downloaded_month_explains_what_to_do(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="spoorcast download --months 2025-12"):
        build_dataset(["2025-12"], settings)
