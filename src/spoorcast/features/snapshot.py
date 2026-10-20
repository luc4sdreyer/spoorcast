"""The serving feature snapshot: aggregates as of one date, plus the station directory.

Written by ``spoorcast snapshot-features`` and read by the API's local ``FeatureLookup``.

Layout::

    <snapshot_dir>/snapshot.json                  as_of date, window, creation time
    <snapshot_dir>/stations.parquet               station_code, station_name, weather_point_id
    <snapshot_dir>/<aggregate name>.parquet       key columns, n, rate
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from spoorcast.features.aggregates import AGGREGATES, DailyCounts, build_snapshot

SNAPSHOT_INFO_FILE = "snapshot.json"
STATIONS_FILE = "stations.parquet"


@dataclass
class FeatureSnapshot:
    info: dict[str, Any]
    stations: pd.DataFrame
    aggregates: dict[str, pd.DataFrame]


def write_snapshot(
    directory: Path,
    daily: DailyCounts,
    stations: pd.DataFrame,
    as_of: date,
    window_days: int,
) -> Path:
    # MLOPS-SEAM: this file-based snapshot is the stand-in for an (online) feature table.
    directory.mkdir(parents=True, exist_ok=True)
    dutch = stations.loc[
        stations["country"] == "NL", ["station_code", "station_name", "weather_point_id"]
    ]
    dutch.reset_index(drop=True).to_parquet(directory / STATIONS_FILE, index=False)
    for name, table in build_snapshot(daily, as_of, window_days).items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
    info = {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "created_at": datetime.now(UTC).isoformat(),
        "n_stations": len(dutch),
    }
    (directory / SNAPSHOT_INFO_FILE).write_text(json.dumps(info, indent=2))
    return directory


def read_snapshot(directory: Path) -> FeatureSnapshot:
    info_path = directory / SNAPSHOT_INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(
            f"No feature snapshot in {directory}; run `spoorcast snapshot-features`"
        )
    return FeatureSnapshot(
        info=json.loads(info_path.read_text()),
        stations=pd.read_parquet(directory / STATIONS_FILE),
        aggregates={
            spec.name: pd.read_parquet(directory / f"{spec.name}.parquet") for spec in AGGREGATES
        },
    )
