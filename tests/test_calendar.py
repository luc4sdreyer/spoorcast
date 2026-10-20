from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from spoorcast.features.calendar import (
    calendar_features,
    dutch_public_holidays,
    easter_sunday,
    hour_bucket,
)
from spoorcast.months import add_months, month_bounds, parse_months


def ts(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values), utc=True))


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5))],
)
def test_easter(year: int, expected: date) -> None:
    assert easter_sunday(year) == expected


def test_kings_day_moves_to_saturday_when_on_sunday() -> None:
    assert date(2025, 4, 26) in dutch_public_holidays(2025)  # 27 April 2025 was a Sunday
    assert date(2026, 4, 27) in dutch_public_holidays(2026)
    assert date(2026, 5, 14) in dutch_public_holidays(2026)  # Ascension Day


def test_local_time_handles_dst() -> None:
    # 06:30 UTC is 07:30 in winter (CET) and 08:30 in summer (CEST).
    features = calendar_features(ts("2026-01-14T06:30:00Z", "2026-07-15T06:30:00Z"))
    assert features["hour"].tolist() == [7, 8]
    # The night of the switch to summer time: 00:30 UTC is 01:30 CET, 01:30 UTC is 03:30 CEST.
    features = calendar_features(ts("2026-03-29T00:30:00Z", "2026-03-29T01:30:00Z"))
    assert features["hour"].tolist() == [1, 3]


def test_weekend_holiday_and_rush_hour() -> None:
    features = calendar_features(
        ts(
            "2026-02-02T07:00:00Z",  # Monday 08:00 local: rush hour
            "2026-02-02T11:00:00Z",  # Monday 12:00: not rush hour
            "2026-02-07T07:00:00Z",  # Saturday 08:00: weekend, no rush hour
            "2026-04-06T07:00:00Z",  # Easter Monday 09:00 CEST: holiday
            "2026-02-02T17:15:00Z",  # Monday 18:15: rush hour
        )
    )
    assert features["is_rush_hour"].tolist() == [1, 0, 0, 0, 1]
    assert features["is_weekend"].tolist() == [0, 0, 1, 0, 0]
    assert features["is_holiday"].tolist() == [0, 0, 0, 1, 0]
    assert features["day_of_week"].tolist() == [0, 0, 5, 0, 0]


def test_hour_bucket() -> None:
    buckets = hour_bucket(
        ts("2026-02-02T03:00:00Z", "2026-02-02T06:00:00Z", "2026-02-02T22:00:00Z")
    )
    assert buckets.tolist() == [0, 1, 4]  # 04:00 night, 07:00 am_peak, 23:00 evening


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        calendar_features(pd.Series(pd.to_datetime(["2026-02-02T07:00:00"])))


def test_months() -> None:
    assert parse_months("2025-11..2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert parse_months("2026-03,2026-01") == ["2026-01", "2026-03"]
    assert add_months("2026-01", -1) == "2025-12"
    assert month_bounds("2024-02") == (date(2024, 2, 1), date(2024, 2, 29))
    with pytest.raises(ValueError, match="YYYY-MM"):
        parse_months("2026-13")
