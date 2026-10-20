"""Point-in-time correct historical aggregates (trailing late rates).

The aggregate for a departure on service date ``D`` only uses departures whose service date
lies in ``[D - window, D - 1]``: strictly earlier than ``D``. Both training (one value per
departure date) and serving (one value as of a snapshot date) go through
:func:`trailing_rates`, so they cannot drift apart.

Pipeline::

    labelled departures --daily_counts--> per-day counts --trailing_rates--> rates per as-of date
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd

from spoorcast.features.calendar import hour_bucket


@dataclass(frozen=True)
class AggregateSpec:
    name: str
    keys: tuple[str, ...]


AGGREGATES: tuple[AggregateSpec, ...] = (
    AggregateSpec("station_late_rate_28d", ("station_code",)),
    AggregateSpec("station_hour_late_rate_28d", ("station_code", "hour_bucket")),
    AggregateSpec("service_type_late_rate_28d", ("service_type",)),
)
AGGREGATE_FEATURES = [spec.name for spec in AGGREGATES]
AGGREGATES_BY_NAME = {spec.name: spec for spec in AGGREGATES}

# Daily counts per aggregate: key columns + service_date, n, n_late.
DailyCounts = Mapping[str, pd.DataFrame]
# Snapshot per aggregate: key columns + rate.
Snapshot = Mapping[str, pd.DataFrame]


def with_keys(inputs: pd.DataFrame) -> pd.DataFrame:
    """The columns every aggregate is keyed on, derived from the canonical inputs."""
    return pd.DataFrame(
        {
            "station_code": inputs["station_code"],
            "service_type": inputs["service_type"],
            "hour_bucket": hour_bucket(inputs["scheduled_departure"]),
        },
        index=inputs.index,
    )


def daily_counts(labelled: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count departures and late departures per key and service date."""
    keyed = with_keys(labelled)
    keyed["service_date"] = labelled["service_date"]
    keyed["late"] = labelled["late"].astype("int64")
    out = {}
    for spec in AGGREGATES:
        grouped = keyed.groupby([*spec.keys, "service_date"], observed=True)["late"]
        out[spec.name] = grouped.agg(n="count", n_late="sum").reset_index()
    return out


def combine_daily_counts(parts: Iterable[DailyCounts]) -> dict[str, pd.DataFrame]:
    """Merge daily counts computed on separate chunks (for example one per month)."""
    parts = list(parts)
    out = {}
    for spec in AGGREGATES:
        frame = pd.concat([part[spec.name] for part in parts], ignore_index=True)
        grouped = frame.groupby([*spec.keys, "service_date"], observed=True)[["n", "n_late"]]
        out[spec.name] = grouped.sum().reset_index()
    return out


def trailing_rates(
    daily: pd.DataFrame, keys: Iterable[str], as_of_dates: Iterable[date], window_days: int
) -> pd.DataFrame:
    """Late rate per key over the ``window_days`` days strictly before each as-of date.

    Returns key columns, ``as_of_date``, ``n`` and ``rate``. Keys without any departure in a
    window are absent (they become missing values when joined).
    """
    keys = list(keys)
    as_of = pd.DataFrame({"as_of_date": pd.to_datetime(sorted(set(as_of_dates)))})
    key_sql = ", ".join(f"d.{k}" for k in keys)
    con = duckdb.connect()
    try:
        con.register("daily", daily)
        con.register("as_of", as_of)
        result = con.execute(
            f"""
            SELECT {key_sql}, a.as_of_date,
                   sum(d.n)::BIGINT AS n,
                   sum(d.n_late)::DOUBLE / sum(d.n) AS rate
            FROM as_of AS a
            JOIN daily AS d
              ON d.service_date <  a.as_of_date
             AND d.service_date >= a.as_of_date - INTERVAL {int(window_days)} DAY
            GROUP BY ALL
            """
        ).df()
    finally:
        con.close()
    result["as_of_date"] = pd.to_datetime(result["as_of_date"]).astype("datetime64[ns]")
    return result


def point_in_time_aggregates(
    inputs: pd.DataFrame, daily: DailyCounts, window_days: int
) -> pd.DataFrame:
    """Aggregates for each training row as of its own ``service_date`` (training path)."""
    keyed = with_keys(inputs)
    keyed["as_of_date"] = pd.to_datetime(inputs["service_date"]).astype("datetime64[ns]")
    dates = keyed["as_of_date"].dt.date.unique()
    out = pd.DataFrame(index=inputs.index)
    for spec in AGGREGATES:
        rates = trailing_rates(daily[spec.name], spec.keys, dates, window_days)
        joined = keyed.merge(
            rates[[*spec.keys, "as_of_date", "rate"]],
            on=[*spec.keys, "as_of_date"],
            how="left",
            validate="many_to_one",
        )
        out[spec.name] = joined["rate"].to_numpy(dtype="float64")
    return out


def build_snapshot(daily: DailyCounts, as_of: date, window_days: int) -> dict[str, pd.DataFrame]:
    """Aggregates as of one date, for serving (written by ``spoorcast snapshot-features``)."""
    snapshot = {}
    for spec in AGGREGATES:
        rates = trailing_rates(daily[spec.name], spec.keys, [as_of], window_days)
        snapshot[spec.name] = rates[[*spec.keys, "n", "rate"]].reset_index(drop=True)
    return snapshot


def lookup_snapshot(inputs: pd.DataFrame, snapshot: Snapshot) -> pd.DataFrame:
    """Aggregates for serving rows from a snapshot (serving path)."""
    keyed = with_keys(inputs)
    out = pd.DataFrame(index=inputs.index)
    for spec in AGGREGATES:
        table = snapshot[spec.name]
        joined = keyed.merge(
            table[[*spec.keys, "rate"]], on=list(spec.keys), how="left", validate="many_to_one"
        )
        out[spec.name] = joined["rate"].to_numpy(dtype="float64")
    return out
