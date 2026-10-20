"""Weather features: the hourly weather at the departure hour, at the nearest weather point.

The departure is matched to the hourly record at the start of its hour (``floor``). For
precipitation and snowfall that record is the sum over the hour *before* that timestamp, so
no weather observed after the departure is ever used.
"""

from __future__ import annotations

import pandas as pd

from spoorcast.data.weather import WEATHER_VARIABLES

WEATHER_FEATURES = list(WEATHER_VARIABLES)


def weather_hour(scheduled_departure: pd.Series) -> pd.Series:
    """The UTC hour (tz-naive) whose weather record applies to a departure."""
    return _naive_utc(scheduled_departure).dt.floor("h")


def _naive_utc(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")


def lookup_weather(inputs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Weather for each input row, from an hourly table.

    ``inputs`` needs ``weather_point_id`` and ``scheduled_departure``; ``table`` is the output
    of :func:`spoorcast.data.weather.parse_open_meteo` (or several concatenated). Rows without
    a matching hour get missing values.
    """
    keys = pd.DataFrame(
        {
            "weather_point_id": inputs["weather_point_id"].to_numpy(),
            "time_utc": weather_hour(inputs["scheduled_departure"]).to_numpy(),
        }
    )
    right = table[["weather_point_id", "time_utc", *WEATHER_FEATURES]].copy()
    right["time_utc"] = _naive_utc(right["time_utc"])
    joined = keys.merge(right, on=["weather_point_id", "time_utc"], how="left", validate="m:1")
    out = joined[WEATHER_FEATURES].astype("float64")
    out.index = inputs.index
    return out
