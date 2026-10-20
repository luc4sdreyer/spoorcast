"""Aggregate features must only use departures from strictly earlier service dates."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from spoorcast.config import Settings
from spoorcast.data.dataset import read_partitions
from spoorcast.features.aggregates import (
    AGGREGATES,
    daily_counts,
    point_in_time_aggregates,
    with_keys,
)

WINDOW = 28


def brute_force_rate(row: pd.Series, history: pd.DataFrame, keys: tuple[str, ...]) -> float:
    """Late rate over departures with the same keys and service date in [D-28, D-1]."""
    day = row["service_date"]
    mask = (history["service_date"] < day) & (history["service_date"] >= day - timedelta(WINDOW))
    for key in keys:
        mask &= history[key] == row[key]
    selected = history.loc[mask, "late"]
    return float(selected.mean()) if len(selected) else np.nan


@pytest.fixture(scope="module")
def departures(built_settings: Settings) -> pd.DataFrame:
    deps = read_partitions(built_settings.departures_dir)
    keyed = with_keys(deps)
    deps["hour_bucket"] = keyed["hour_bucket"]
    return deps


def test_every_aggregate_matches_a_strictly_earlier_brute_force(
    built_settings: Settings, departures: pd.DataFrame
) -> None:
    dataset = read_partitions(built_settings.dataset_dir)
    merged = dataset.merge(
        departures[["service_id", "station_code", "scheduled_departure", "hour_bucket"]],
        on=["service_id", "station_code", "scheduled_departure"],
        validate="one_to_one",
    )
    assert len(merged) == len(dataset) > 0
    for spec in AGGREGATES:
        expected = merged.apply(brute_force_rate, axis=1, args=(departures, spec.keys))
        np.testing.assert_allclose(merged[spec.name], expected, equal_nan=True, err_msg=spec.name)
    # The very first service date has no history at all.
    first_day = merged["service_date"] == merged["service_date"].min()
    assert merged.loc[first_day, [s.name for s in AGGREGATES]].isna().all().all()


def test_same_day_and_future_outcomes_do_not_change_features(departures: pd.DataFrame) -> None:
    day = departures["service_date"].sort_values().unique()[3]
    rows = departures[departures["service_date"] == day]
    before = point_in_time_aggregates(rows, daily_counts(departures), WINDOW)

    # Flip every outcome on that day and later: features of that day must not move.
    tampered = departures.copy()
    future = tampered["service_date"] >= day
    tampered.loc[future, "late"] = 1 - tampered.loc[future, "late"]
    after = point_in_time_aggregates(rows, daily_counts(tampered), WINDOW)
    pd.testing.assert_frame_equal(before, after)
    assert before.notna().all().all()

    # Sanity check that the test can fail: flipping the previous day does move them.
    past = tampered["service_date"] == day - pd.Timedelta(days=1)
    tampered.loc[past, "late"] = 1 - tampered.loc[past, "late"]
    changed = point_in_time_aggregates(rows, daily_counts(tampered), WINDOW)
    assert not before.equals(changed)
