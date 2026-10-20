"""Loading and cleaning of the raw archives into labelled departures.

One row of output is one scheduled departure of one service from one Dutch station. The
columns are the canonical model inputs (see :mod:`spoorcast.features.build`) plus ids, the
observed delay and the ``late`` label.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from spoorcast.data.validate import (
    OPTIONAL_SERVICE_COLUMNS,
    REQUIRED_SERVICE_COLUMNS,
    DataValidationError,
    check_columns,
    validate_departures,
    validate_stations,
)
from spoorcast.data.weather import nearest_point

logger = logging.getLogger(__name__)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_departures(path: Path) -> pd.DataFrame:
    """Read one services archive (gzipped CSV, one row per stop) into departures.

    Keeps rows with a scheduled departure time. ``service_start_time`` is the first scheduled
    departure of the whole service, computed before any stop is filtered out.
    """
    source = path.name
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")
        csv = f"read_csv({_quote(str(path))}, header = true"
        columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {csv})").fetchall()]
        check_columns(columns, REQUIRED_SERVICE_COLUMNS, source)
        unknown = set(columns) - set(REQUIRED_SERVICE_COLUMNS) - OPTIONAL_SERVICE_COLUMNS
        if unknown:
            logger.info("%s: ignoring unexpected column(s) %s", source, sorted(unknown))

        types = ", ".join(f"{_quote(c)}: {_quote(t)}" for c, t in REQUIRED_SERVICE_COLUMNS.items())
        query = f"""
            SELECT
                "Service:RDT-ID"           AS service_id,
                "Service:Date"             AS service_date,
                "Service:Type"             AS service_type,
                "Service:Company"          AS company,
                "Service:Train number"     AS train_number,
                "Stop:Station code"        AS station_code,
                "Stop:Departure time"      AS scheduled_departure,
                service_start_time,
                "Stop:Departure delay"     AS departure_delay_min,
                "Stop:Departure cancelled" AS departure_cancelled
            FROM (
                SELECT
                    *,
                    min("Stop:Departure time") OVER (PARTITION BY "Service:RDT-ID")
                        AS service_start_time
                FROM {csv}, types = {{{types}}})
            )
            WHERE "Stop:Departure time" IS NOT NULL
        """
        try:
            df = con.execute(query).df()
        except duckdb.Error as exc:
            raise DataValidationError(f"{source}: could not parse file: {exc}") from exc
    finally:
        con.close()

    df["service_date"] = pd.to_datetime(df["service_date"]).astype("datetime64[ns]")
    for column in ("scheduled_departure", "service_start_time"):
        df[column] = pd.to_datetime(df[column], utc=True)
    return df


def load_stations(path: Path) -> pd.DataFrame:
    """Load the station list and map each station to its nearest weather point."""
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    validate_stations(raw, path.name)
    stations = pd.DataFrame(
        {
            "station_code": raw["code"].astype(str).str.upper(),
            "station_name": raw["name_long"].astype(str),
            "country": raw["country"].astype(str),
            "lat": raw["geo_lat"].astype("float64"),
            "lon": raw["geo_lng"].astype("float64"),
        }
    )
    stations["weather_point_id"] = [
        nearest_point(lat, lon) for lat, lon in zip(stations["lat"], stations["lon"], strict=True)
    ]
    return stations


def is_passenger_train(service_type: pd.Series) -> pd.Series:
    """Rail replacement buses ("Stopbus ipv trein", "Bus", ...) are not trains."""
    return ~service_type.fillna("").str.contains("bus", case=False)


@dataclass
class CleaningReport:
    source: str
    rows_in: int
    dropped: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0

    def summary(self) -> str:
        drops = ", ".join(f"{reason}={n}" for reason, n in self.dropped.items())
        return f"{self.source}: {self.rows_in} departures in, {self.rows_out} out ({drops})"


def clean_departures(
    departures: pd.DataFrame,
    stations: pd.DataFrame,
    *,
    late_threshold_min: int,
    max_plausible_delay_min: int,
    source: str = "departures",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate, apply the v1 exclusions and add the label. See ``docs/decisions.md``."""
    validate_departures(departures, source, max_plausible_delay_min)
    report = CleaningReport(source=source, rows_in=len(departures))
    dutch = stations.loc[stations["country"] == "NL", ["station_code", "weather_point_id"]]

    df = departures
    filters: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("cancelled", lambda d: ~d["departure_cancelled"].fillna(False).astype(bool)),
        ("not_passenger_train", lambda d: is_passenger_train(d["service_type"])),
        ("not_dutch_station", lambda d: d["station_code"].isin(dutch["station_code"])),
        ("missing_delay", lambda d: d["departure_delay_min"].notna()),
        (
            "implausible_delay",
            lambda d: d["departure_delay_min"].between(0, max_plausible_delay_min),
        ),
        ("missing_category", lambda d: d["service_type"].notna() & d["company"].notna()),
    ]
    for reason, keep in filters:
        mask = keep(df)
        report.dropped[reason] = int((~mask).sum())
        df = df[mask]

    df = df.merge(dutch, on="station_code", how="left", validate="many_to_one")
    df = df.drop(columns=["departure_cancelled"])
    df["departure_delay_min"] = df["departure_delay_min"].astype("int64")
    df["late"] = (df["departure_delay_min"] >= late_threshold_min).astype("int8")
    df["year_month"] = df["service_date"].dt.strftime("%Y-%m")
    df = df.sort_values(["scheduled_departure", "station_code", "service_id"], kind="stable")
    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    return df, report
