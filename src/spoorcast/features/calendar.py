"""Calendar features, computed in local Dutch time (``Europe/Amsterdam``, DST-aware)."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from spoorcast.config import LOCAL_TIMEZONE

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
]

# Hour buckets used as a key for the trailing aggregates (local hour -> bucket).
HOUR_BUCKETS: dict[str, range] = {
    "night": range(0, 6),
    "am_peak": range(6, 9),
    "midday": range(9, 16),
    "pm_peak": range(16, 19),
    "evening": range(19, 24),
}
_HOUR_TO_BUCKET = np.array(
    [next(i for i, hours in enumerate(HOUR_BUCKETS.values()) if h in hours) for h in range(24)]
)

# NS peak hours (weekdays): 06:30-09:00 and 16:00-18:30.
_RUSH_WINDOWS_MIN = ((6 * 60 + 30, 9 * 60), (16 * 60, 18 * 60 + 30))


def easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=64)
def dutch_public_holidays(year: int) -> frozenset[date]:
    """Dutch national public holidays. Liberation Day is included every year for simplicity."""
    easter = easter_sunday(year)
    kings_day = date(year, 4, 27)
    if kings_day.weekday() == 6:  # moved to Saturday when it falls on a Sunday
        kings_day -= timedelta(days=1)
    return frozenset(
        {
            date(year, 1, 1),
            easter - timedelta(days=2),  # Good Friday
            easter,
            easter + timedelta(days=1),
            kings_day,
            date(year, 5, 5),
            easter + timedelta(days=39),  # Ascension Day
            easter + timedelta(days=49),  # Whit Sunday
            easter + timedelta(days=50),  # Whit Monday
            date(year, 12, 25),
            date(year, 12, 26),
        }
    )


def to_local(timestamps: pd.Series) -> pd.Series:
    """Convert tz-aware timestamps to Dutch local time."""
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"Expected tz-aware timestamps, got {timestamps.dtype}")
    return timestamps.dt.tz_convert(LOCAL_TIMEZONE)


def is_holiday(local: pd.Series) -> pd.Series:
    days = local.dt.tz_localize(None).dt.normalize()
    holidays = set().union(*(dutch_public_holidays(int(y)) for y in local.dt.year.unique()))
    return days.isin(pd.to_datetime(sorted(holidays)))


def hour_bucket(scheduled_departure: pd.Series) -> pd.Series:
    """Index of the local-hour bucket (see :data:`HOUR_BUCKETS`) of each departure."""
    hours = to_local(scheduled_departure).dt.hour.to_numpy()
    return pd.Series(_HOUR_TO_BUCKET[hours], index=scheduled_departure.index, dtype="int64")


def calendar_features(scheduled_departure: pd.Series) -> pd.DataFrame:
    local = to_local(scheduled_departure)
    weekday = local.dt.dayofweek
    weekend = weekday >= 5
    holiday = is_holiday(local)
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    in_rush_window = pd.Series(False, index=local.index)
    for start, end in _RUSH_WINDOWS_MIN:
        in_rush_window |= (minute_of_day >= start) & (minute_of_day < end)
    return pd.DataFrame(
        {
            "hour": local.dt.hour.astype("int64"),
            "day_of_week": weekday.astype("int64"),
            "month": local.dt.month.astype("int64"),
            "is_weekend": weekend.astype("int64"),
            "is_holiday": holiday.astype("int64"),
            "is_rush_hour": (in_rush_window & ~weekend & ~holiday).astype("int64"),
        },
        index=scheduled_departure.index,
    )
