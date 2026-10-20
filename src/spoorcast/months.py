"""Helpers for the ``YYYY-MM`` month strings used throughout the pipeline."""

from __future__ import annotations

import re
from datetime import date, timedelta

_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


def parse_month(value: str) -> date:
    """Parse ``YYYY-MM`` into the first day of that month."""
    match = _MONTH_RE.match(value.strip())
    if not match or not 1 <= int(match.group(2)) <= 12:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM")
    return date(int(match.group(1)), int(match.group(2)), 1)


def format_month(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def add_months(month: str, n: int) -> str:
    first = parse_month(month)
    index = first.year * 12 + first.month - 1 + n
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of months from ``start`` to ``end``."""
    if parse_month(start) > parse_month(end):
        raise ValueError(f"Month range {start}..{end} is empty")
    months = [start]
    while months[-1] != end:
        months.append(add_months(months[-1], 1))
    return months


def parse_months(spec: str) -> list[str]:
    """Parse ``2025-09..2026-08``, ``2025-09,2025-11`` or a single ``2026-02``."""
    months: list[str] = []
    for part in spec.split(","):
        if ".." in part:
            start, end = part.split("..", 1)
            months.extend(month_range(start.strip(), end.strip()))
        else:
            months.append(format_month(parse_month(part)))
    return sorted(set(months))


def month_bounds(month: str) -> tuple[date, date]:
    """First and last day of a month."""
    first = parse_month(month)
    last = parse_month(add_months(month, 1)) - timedelta(days=1)
    return first, last
