"""Weather: representative weather points, Open-Meteo downloads and parsing.

We do not fetch weather per railway station. Each station is mapped to the nearest of a
small set of weather points spread over the Netherlands, and hourly series are fetched and
cached per point and per month.

Weather data by Open-Meteo.com (https://open-meteo.com/), licensed under CC BY 4.0.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from spoorcast.config import OPEN_METEO_ARCHIVE_LAG_DAYS, OPEN_METEO_ARCHIVE_URL, Settings
from spoorcast.data.sources import client_scope
from spoorcast.months import format_month, month_bounds, month_range

logger = logging.getLogger(__name__)

# Hourly Open-Meteo variables. Values are hourly: temperature and wind are instantaneous at
# the timestamp, precipitation and snowfall are sums over the *preceding* hour.
WEATHER_VARIABLES = (
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
)


@dataclass(frozen=True)
class WeatherPoint:
    point_id: str
    lat: float
    lon: float


# Twenty points that roughly cover the Dutch rail network, including the coast and borders.
WEATHER_POINTS: tuple[WeatherPoint, ...] = (
    WeatherPoint("amsterdam", 52.379, 4.900),
    WeatherPoint("alkmaar", 52.638, 4.740),
    WeatherPoint("den_helder", 52.955, 4.761),
    WeatherPoint("haarlem", 52.388, 4.638),
    WeatherPoint("den_haag", 52.080, 4.325),
    WeatherPoint("rotterdam", 51.925, 4.469),
    WeatherPoint("vlissingen", 51.444, 3.597),
    WeatherPoint("roosendaal", 51.541, 4.458),
    WeatherPoint("breda", 51.595, 4.780),
    WeatherPoint("utrecht", 52.089, 5.110),
    WeatherPoint("amersfoort", 52.153, 5.373),
    WeatherPoint("den_bosch", 51.690, 5.294),
    WeatherPoint("eindhoven", 51.443, 5.481),
    WeatherPoint("venlo", 51.364, 6.171),
    WeatherPoint("maastricht", 50.850, 5.706),
    WeatherPoint("nijmegen", 51.843, 5.854),
    WeatherPoint("arnhem", 51.985, 5.899),
    WeatherPoint("zwolle", 52.505, 6.091),
    WeatherPoint("enschede", 52.222, 6.890),
    WeatherPoint("leeuwarden", 53.196, 5.793),
    WeatherPoint("groningen", 53.211, 6.564),
)
WEATHER_POINTS_BY_ID = {p.point_id: p for p in WEATHER_POINTS}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_point(lat: float, lon: float, points: Sequence[WeatherPoint] = WEATHER_POINTS) -> str:
    return min(points, key=lambda p: haversine_km(lat, lon, p.lat, p.lon)).point_id


# --- Download -----------------------------------------------------------------------------------


def weather_path(settings: Settings, point_id: str, month: str) -> Path:
    return settings.weather_raw_dir / point_id / f"{month}.json"


def _month_is_final(month: str, today: date) -> bool:
    _, last = month_bounds(month)
    return last + timedelta(days=OPEN_METEO_ARCHIVE_LAG_DAYS) < today


def download_weather(
    points: Iterable[WeatherPoint],
    start: date,
    end: date,
    settings: Settings,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> list[Path]:
    """Fetch hourly archive weather per point and month, caching one JSON file each.

    Months that the archive may not have fully published yet are fetched but not treated as
    cached, so a later run refreshes them.
    """
    today = today or date.today()
    months = month_range(format_month(start), format_month(end))
    paths: list[Path] = []
    with client_scope(client) as http:
        for point in points:
            for month in months:
                path = weather_path(settings, point.point_id, month)
                final = _month_is_final(month, today)
                if path.exists() and final:
                    paths.append(path)
                    continue
                first, last = month_bounds(month)
                last = min(last, today - timedelta(days=1))
                if last < first:
                    continue
                payload = _fetch_archive(http, point, first, last)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".json.part")
                tmp.write_text(json.dumps(payload))
                tmp.rename(path)
                logger.info("Downloaded weather %s %s", point.point_id, month)
                paths.append(path)
    return paths


def _fetch_archive(
    client: httpx.Client, point: WeatherPoint, first: date, last: date
) -> dict[str, Any]:
    response = client.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": point.lat,
            "longitude": point.lon,
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "GMT",
            "wind_speed_unit": "kmh",
        },
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


# --- Parsing ------------------------------------------------------------------------------------


def parse_open_meteo(payload: dict[str, Any], point_id: str) -> pd.DataFrame:
    """Turn an Open-Meteo hourly response (GMT timestamps) into a tidy frame.

    Columns: ``weather_point_id``, ``time_utc`` (tz-aware hour) and one per variable. Works for
    both archive and forecast responses.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"Open-Meteo response for {point_id} has no hourly data")
    frame = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"], utc=True)})
    for variable in WEATHER_VARIABLES:
        frame[variable] = pd.to_numeric(pd.Series(hourly.get(variable)), errors="coerce").astype(
            "float64"
        )
    frame.insert(0, "weather_point_id", point_id)
    return frame


def load_weather_table(weather_dir: Path, months: Iterable[str] | None = None) -> pd.DataFrame:
    """Load every cached archive file (optionally only some months) into one table."""
    wanted = set(months) if months is not None else None
    frames = [
        parse_open_meteo(json.loads(path.read_text()), path.parent.name)
        for path in sorted(weather_dir.glob("*/*.json"))
        if wanted is None or path.stem in wanted
    ]
    if not frames:
        raise FileNotFoundError(f"No cached weather files found under {weather_dir}")
    table = pd.concat(frames, ignore_index=True)
    return table.drop_duplicates(["weather_point_id", "time_utc"], keep="last").reset_index(
        drop=True
    )
