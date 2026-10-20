"""Replay a month of real departures against the API (the drift-demo driver).

This is a client only: it sends requests in schedule order and records the actual outcome of
each prediction in ``replay_labels.jsonl``. It computes no metrics and no drift.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from spoorcast.config import Settings
from spoorcast.data.dataset import read_partitions

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    sent: int
    labels_path: Path
    weather_provider: str | None


def load_replay_departures(settings: Settings, month: str, limit: int | None) -> pd.DataFrame:
    departures = read_partitions(settings.departures_dir, [month])
    departures = departures.sort_values(
        ["scheduled_departure", "station_code", "service_id"], kind="stable"
    )
    if limit is not None:
        departures = departures.head(limit)
    return departures.reset_index(drop=True)


def _instance(row: pd.Series) -> dict[str, str | None]:
    start = row["service_start_time"]
    return {
        "station_code": row["station_code"],
        "scheduled_departure": row["scheduled_departure"].isoformat(),
        "service_type": row["service_type"],
        "company": row["company"],
        "service_start_time": None if pd.isna(start) else start.isoformat(),
    }


def replay(
    departures: pd.DataFrame,
    client: httpx.Client,
    labels_path: Path,
    *,
    batch_size: int = 100,
    speedup: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayResult:
    """Send ``departures`` in schedule order and write one label line per prediction.

    With ``speedup`` the schedule is replayed in (scaled) real time: a departure planned one
    hour after the first is sent ``3600 / speedup`` seconds after the start. Without it,
    batches are sent back to back.
    """
    weather_provider = None
    info = client.get("/v1/model")
    if info.status_code == 200:
        weather_provider = info.json().get("weather_provider")
        if weather_provider != "historical":
            logger.warning(
                "The API uses the %r weather provider; start it with "
                "SPOORCAST_WEATHER_PROVIDER=historical for a faithful replay",
                weather_provider,
            )
    else:
        info.raise_for_status()

    # MLOPS-SEAM: these labels are the ground truth to join with the inference log
    # (on prediction_id) for realised-performance and drift monitoring.
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    sent = 0
    first = departures["scheduled_departure"].iloc[0] if len(departures) else None
    wall_start = time.monotonic()
    with labels_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(departures), batch_size):
            chunk = departures.iloc[start : start + batch_size]
            if speedup:
                offset = (chunk["scheduled_departure"].iloc[0] - first).total_seconds() / speedup
                wait = wall_start + offset - time.monotonic()
                if wait > 0:
                    sleep(wait)
            response = client.post(
                "/v1/predictions/batch",
                json={"instances": [_instance(row) for _, row in chunk.iterrows()]},
            )
            response.raise_for_status()
            predictions = response.json()["predictions"]
            for (_, row), prediction in zip(chunk.iterrows(), predictions, strict=True):
                label = {
                    "prediction_id": prediction["prediction_id"],
                    "late": bool(row["late"]),
                    "departure_delay_min": int(row["departure_delay_min"]),
                    "service_id": int(row["service_id"]),
                    "station_code": row["station_code"],
                    "scheduled_departure": row["scheduled_departure"].isoformat(),
                }
                out.write(json.dumps(label) + "\n")
            sent += len(chunk)
            logger.info("Replayed %d/%d departures", sent, len(departures))
    return ReplayResult(sent=sent, labels_path=labels_path, weather_provider=weather_provider)
