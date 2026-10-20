# Spoorcast

Spoorcast predicts the probability that a Dutch passenger train **departs at least 5 minutes
late** from a given station. It uses the timetable, the calendar, the weather and recent
punctuality at that station.

This repository holds the ML application: data download and preparation, feature
engineering, training and evaluation, a model wrapper, a FastAPI service and a replay tool.
The MLOps platform around it (experiment tracking, registry, serving, monitoring, retraining
on Azure Databricks) plugs into documented seams; see [`docs/mlops-seams.md`](docs/mlops-seams.md).
Modelling and data decisions are recorded in [`docs/decisions.md`](docs/decisions.md).

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make sync                                   # create .venv from uv.lock
make test lint                              # offline; uses the fixture in tests/fixtures

uv run spoorcast download --months 2025-09..2026-08     # ~400 MB, cached and resumable
uv run spoorcast build-dataset --months 2025-09..2026-08  # ~2 min
uv run spoorcast train --validate-month 2026-08           # ~1 min, saves models/latest
uv run spoorcast snapshot-features                         # aggregates for serving
make serve                                                 # http://127.0.0.1:8000/docs
```

```bash
curl -s -X POST http://127.0.0.1:8000/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"station_code": "UT", "scheduled_departure": "2026-10-21T08:12:00+02:00",
       "service_type": "Intercity", "company": "NS",
       "service_start_time": "2026-10-21T07:35:00+02:00"}'
# {"prediction_id":"…","probability_late":0.04,"late":false,"model_version":"20261020T…"}
```

## How it works

```
 Rijden de Treinen archive ─┐                       ┌─ point-in-time aggregates (28 days,
 (one row per stop)         │                       │  strictly before the service date)
                            ├─ ingest + validate ───┤
 Station list ──────────────┤  (departures, label)  ├─ calendar (local time, holidays, peak)
                            │                       │
 Open-Meteo archive ────────┘                       └─ weather at the departure hour
 (21 weather points)                                        │
                                                            ▼
                                              build_features()  ◄── also used by the API
                                                            │
                                     LightGBM vs. station×hour baseline (time-based split)
```

* **Unit of prediction:** one scheduled departure of one service from one Dutch station.
  Label: departure delay ≥ 5 minutes (`SPOORCAST_LATE_THRESHOLD_MIN`). Cancelled
  departures, rail-replacement buses and foreign stations are excluded.
* **Features** (`src/spoorcast/features/`): station, service type, operator; local hour, day
  of week, month, weekend, Dutch public holiday, peak hours; minutes since the service
  started; temperature, precipitation, snowfall, wind speed and gusts at the nearest of 21
  weather points; trailing 28-day late rates for the station, the station × hour bucket and
  the service type.
* **One feature function for training and serving:** `build_features()` takes the canonical
  inputs plus looked-up weather and aggregates. Training looks those up from the archive;
  the API looks them up through a `WeatherProvider` and a `FeatureLookup`. A parity test
  checks that both paths give identical features.
* **No leakage:** aggregates use only departures with a strictly earlier service date; a test
  checks this against a brute-force computation and by tampering with same-day outcomes.
  Weather is taken at the start of the departure hour.
* **Time-based split:** train on months `[t-12, t-1]`, validate on month `t`.

### Results (validate on 2026-08, trained on 2025-09..2026-07)

| | ROC AUC | PR AUC | Brier | Log loss |
|---|---|---|---|---|
| LightGBM | 0.717 | 0.138 | 0.0346 | 0.148 |
| Baseline (station × hour late rate) | 0.651 | 0.061 | 0.0362 | 0.157 |

Base rate in August 2026: 3.8% of departures were 5+ minutes late. See
[`docs/decisions.md`](docs/decisions.md#results) for the calibration tables.

## CLI

| Command | What it does |
|---|---|
| `spoorcast download --months 2025-09..2026-08` | Services (monthly), stations, hourly weather. Cached. |
| `spoorcast build-dataset --months …` | Clean, label, featurise → `data/processed/dataset/year_month=…` |
| `spoorcast train --validate-month 2026-08` | Train, compare with the baseline, save to `models/latest` |
| `spoorcast evaluate --month 2026-08` | Score a saved model and the baseline on a month |
| `spoorcast snapshot-features [--as-of DATE]` | Aggregates snapshot for the API |
| `spoorcast serve` | Run the API |
| `spoorcast replay --month 2026-02 --limit 1000` | Replay real departures against the API |

The `Makefile` has the same targets plus `lint`, `format`, `typecheck`, `test`,
`docker-build`, `docker-run` and `serve-replay`. `make help` lists them.

## API

| Endpoint | |
|---|---|
| `POST /v1/predictions` | One departure. `?debug=true` adds `features_used`. |
| `POST /v1/predictions/batch` | `{"instances": [...]}`, up to 500, order preserved. |
| `GET /health` | Liveness. |
| `GET /ready` | 200 when the model and the feature snapshot are loaded, else 503. |
| `GET /v1/model` | Model version, metadata, decision threshold, snapshot date. |

Request fields: `station_code`, `scheduled_departure` (RFC 3339 with offset), `service_type`,
`company`, and optionally `service_start_time` (the service's first departure). Unknown
stations and timestamps without an offset return 422. Every response carries an
`X-Request-ID` header. Logs are JSON lines on stderr. Each prediction is written to the
inference log (stdout, or `SPOORCAST_INFERENCE_LOG_PATH`).

Configuration (environment variables or `.env`):

| Variable | Default | |
|---|---|---|
| `SPOORCAST_MODEL_DIR` | `models/latest` | Saved `SpoorcastModel` |
| `SPOORCAST_SNAPSHOT_DIR` | `data/processed/feature_snapshot` | Aggregates snapshot |
| `SPOORCAST_WEATHER_PROVIDER` | `open-meteo` | `open-meteo` (forecast), `historical` (cached archive), `fixed` |
| `SPOORCAST_DECISION_THRESHOLD` | `0.2` | `late` is `probability_late >= threshold` |
| `SPOORCAST_INFERENCE_LOG_PATH` | stdout | JSON lines, one per prediction |
| `SPOORCAST_DATA_DIR` | `data` | Raw and processed data |
| `SPOORCAST_LATE_THRESHOLD_MIN` | `5` | Label threshold (dataset build) |

### Docker

```bash
make docker-build
make docker-run      # mounts models/latest and data/processed/feature_snapshot read-only
```

The image is `python:3.12-slim` and runs as a non-root user. The model directory is mounted
at `/models/model` and the snapshot at `/models/feature_snapshot`.

## Replay (drift demo driver)

`spoorcast replay` sends a month of real departures to the API in schedule order and writes
`replay_labels.jsonl` (`prediction_id` → actual outcome). For a faithful replay the API must
use the observed weather and a snapshot as of the start of that month:

```bash
make serve-replay REPLAY_MONTH=2026-02      # snapshot as of 2026-02-01, historical weather
make replay REPLAY_MONTH=2026-02 REPLAY_LIMIT=1000      # in another terminal
# or: uv run spoorcast replay --month 2026-02 --limit 1000 --speedup 600
```

`--speedup 600` replays the timetable at 600× real time (an hour of departures in six
seconds). Without it, batches are sent back to back. The replay computes no metrics; join
`replay_labels.jsonl` with the inference log on `prediction_id`.

## Laptop memory

The dataset is built one month at a time (peak ≈ 1.5 GB). Training samples 3 million rows
uniformly from the ~16 million in the training window (`--max-train-rows`, `0` = all); the
peak is about 2.5 GB. See `docs/decisions.md`.

## Repository layout

```
src/spoorcast/
├── config.py          settings (SPOORCAST_* env vars) and source URLs
├── months.py          YYYY-MM helpers
├── cli.py             Typer CLI
├── data/              sources.py (downloads), ingest.py, validate.py, weather.py, dataset.py
├── features/          build.py (build_features), calendar.py, aggregates.py, weather.py, snapshot.py
├── training/          train.py, evaluate.py, baseline.py
├── model.py           SpoorcastModel (preprocessing + LightGBM)
├── serving/           app.py, schemas.py, providers.py, predict.py, logging.py
└── replay.py          replay client
tests/                 pytest suite + fixtures/ (real-shaped extract, see scripts/make_fixtures.py)
```

## Data sources and attribution

* **Train services and stations:** [Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data)
  open data (train archive and station list), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* **Weather:** [Weather data by Open-Meteo.com](https://open-meteo.com/), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The historical data come from
  the Open-Meteo Historical Weather API and the serving data from the Forecast API. The free
  API is for non-commercial use.

Data are downloaded to `data/` (git-ignored). The small extract in `tests/fixtures/` is derived
from the same sources under the same licences.

## License

Code: MIT (see `LICENSE`).
