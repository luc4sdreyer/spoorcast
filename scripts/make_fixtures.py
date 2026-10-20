"""Regenerate the committed test fixtures from downloaded real data.

Usage (after ``spoorcast download --months 2026-01..2026-02``)::

    uv run python scripts/make_fixtures.py

Writes a small extract to ``tests/fixtures``: stops at two stations (Gouda, Woerden) for the
service dates 2026-01-27..2026-02-03, the matching stations, and trimmed weather files.
Late departures are over-sampled so the tiny fixture contains both classes.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import duckdb

from spoorcast.config import Settings
from spoorcast.data.ingest import load_stations

STATIONS = ("GD", "WD")
EXTRA_STATIONS = ("UT", "EMN")  # a known station without rows and a foreign one
FIRST_DATE, LAST_DATE = "2026-01-27", "2026-02-03"
PER_DAY_AND_STATION = 30
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    settings = Settings()
    raw = settings.services_raw_dir
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    for month in ("2026-01", "2026-02"):
        path = raw / f"services-{month}.csv.gz"
        con.execute(
            f"""
            CREATE OR REPLACE TABLE stops AS
            SELECT * FROM read_csv('{path}', header = true, all_varchar = true)
            WHERE "Service:Date" BETWEEN '{FIRST_DATE}' AND '{LAST_DATE}'
              AND "Stop:Station code" IN {STATIONS}
            """
        )
        # Per day and station: every late departure plus a seeded random sample of the rest.
        rows = con.execute(
            f"""
            WITH ranked AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY "Service:Date", "Stop:Station code"
                           ORDER BY (TRY_CAST("Stop:Departure delay" AS INT) >= 5) DESC,
                                    hash("Stop:RDT-ID")
                       ) AS rn
                FROM stops
            )
            SELECT * EXCLUDE (rn) FROM ranked
            WHERE rn <= {PER_DAY_AND_STATION}
            ORDER BY "Service:Date", "Stop:Departure time", "Stop:RDT-ID"
            """
        ).fetchall()
        columns = [d[0] for d in con.description]
        if not rows:
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        with gzip.open(OUT / "services" / f"services-{month}.csv.gz", "wt", newline="") as fh:
            fh.write(",".join(columns) + "\n")
            for row in rows:
                fh.write(",".join(_csv(v) for v in row) + "\n")
        print(f"{month}: {len(rows)} rows")

    stations_csv = settings.stations_raw_path.read_text().splitlines()
    keep = set(STATIONS) | set(EXTRA_STATIONS)
    subset = [stations_csv[0]] + [
        line for line in stations_csv[1:] if line.split(",")[1].strip('"') in keep
    ]
    (OUT / "stations").mkdir(parents=True, exist_ok=True)
    (OUT / "stations" / settings.stations_raw_path.name).write_text("\n".join(subset) + "\n")

    points = set(
        load_stations(settings.stations_raw_path)
        .set_index("station_code")
        .loc[list(STATIONS), "weather_point_id"]
    )
    for point in sorted(points):
        for month in ("2026-01", "2026-02"):
            payload = json.loads((settings.weather_raw_dir / point / f"{month}.json").read_text())
            hourly = payload["hourly"]
            keep_idx = [
                i for i, t in enumerate(hourly["time"]) if "2026-01-20" <= t[:10] <= "2026-02-05"
            ]
            payload["hourly"] = {k: [v[i] for i in keep_idx] for k, v in hourly.items()}
            target = OUT / "weather" / point / f"{month}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload))
    print(f"weather points: {sorted(points)}")


def _csv(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return f'"{text}"' if "," in text or '"' in text else text


if __name__ == "__main__":
    main()
