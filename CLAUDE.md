# Spoorcast — Agent Brief

> Drop this file into the repo root as `AGENTS.md` / `CLAUDE.md` (or reference it from there).
> It describes what to build, what **not** to build, and where to leave seams.

## 1. What Spoorcast is

Spoorcast predicts the probability that a Dutch passenger train will **depart late** from a given station, from the schedule, calendar and weather.

It is a portfolio project. Its purpose is to demonstrate **production-grade MLOps on Azure Databricks**: experiment tracking, model registry, serving, CI/CD, monitoring, drift detection and retraining. The model itself only needs to be sensible. A plain gradient-boosted model is the target, and leaderboard accuracy is not a goal.

**Ownership split (important):**

- **You (the agent)** build the "ordinary" ML application: data download and prep, feature engineering, training and evaluation code, a model wrapper, a FastAPI service, a replay/load tool, tests, Dockerfile.
- **The human (Lucas)** builds every MLOps part himself, as a learning exercise. You must **not** implement those parts (see §8). Instead, leave clean, documented seams so they can be plugged in later.

## 2. Tech constraints

| Area | Choice |
|---|---|
| Language | Python 3.12 (keep compatible with a current Databricks Runtime ML) |
| Packaging | `pyproject.toml`, `src/` layout, buildable as a wheel (hatchling). Use `uv` for env/lock |
| Data prep | DuckDB and/or Polars locally; hand pandas DataFrames to the model |
| Model | LightGBM binary classifier |
| API | FastAPI + Pydantic v2 + uvicorn |
| CLI | Typer |
| HTTP client | httpx |
| Quality | ruff (lint + format), pytest, type hints throughout (mypy-clean is a bonus, not a blocker) |
| Container | Dockerfile for the API (slim base, non-root user) |

- Ask before adding dependencies beyond these plus their obvious transitive needs.
- **Do not import `mlflow`, `databricks-sdk`, `pyspark` or any Azure SDK anywhere.**

## 3. Data sources

### 3.1 Train services: Rijden de Treinen train archive
- Docs: https://www.rijdendetreinen.nl/en/open-data/train-archive
- License: **CC BY 4.0**, so the README must credit "Rijden de Treinen" as the source.
- Gzipped CSV, **one row per stop** of a service. Data since 2019.
- Monthly files: `https://opendata.rijdendetreinen.nl/public/services/services-YYYY-MM.csv.gz` (~30 MB each).
- Yearly files: `.../services-YYYY.csv.gz` (~350–380 MB). Monthly files are preferred because they mirror how new data "arrives".
- Mirror (useful if the primary is slow): `https://blobs.duckdb.org/nl-railway/services-YYYY-MM.csv.gz`.

Columns:
`Service:RDT-ID, Service:Date, Service:Type, Service:Company, Service:Train number, Service:Completely cancelled, Service:Partly cancelled, Service:Maximum delay, Stop:RDT-ID, Stop:Station code, Stop:Station name, Stop:Arrival time, Stop:Arrival delay, Stop:Arrival cancelled, Stop:Departure time, Stop:Departure delay, Stop:Departure cancelled, Stop:Platform change, Stop:Planned platform, Stop:Actual platform`

Notes:
- Times are RFC 3339 with offset. Convert to `Europe/Amsterdam` for calendar features, and mind DST.
- `Service:Date` is the schedule date, which can differ from the actual date for services crossing midnight.
- The newer files may contain platform columns that older files lack. Handle a missing column gracefully.

### 3.2 Stations
- https://www.rijdendetreinen.nl/en/open-data/stations. This gives station code, name, lat/lon and category. Get the download URL from that page.

### 3.3 Weather
- Preferred: the **Open-Meteo** historical archive API (hourly, no API key) for training, and the Open-Meteo forecast API for serving. Verify the current endpoints and terms, and add the attribution they require.
- Optional alternative: KNMI Data Platform (requires a free API key and per-timestamp files, which is slow for long ranges). Do not use it unless Open-Meteo is unsuitable.
- Do not fetch weather per railway station (~400 stations). Map each station to the nearest of a small set (~15–25) of representative weather points covering the Netherlands, then fetch and cache hourly series per point.
- Variables: temperature, precipitation, snowfall, wind speed, wind gusts. Add others only if cheap.

### 3.4 Local storage
- `data/raw/` holds downloaded files as-is, cached and skipped if already present.
- `data/processed/` holds Parquet output.
- Both are git-ignored.
- Every download is idempotent and resumable.

## 4. Prediction task

- **Unit of prediction:** one scheduled departure of one service from one station (rows where `Stop:Departure time` is not null).
- **Label `late`:** `Stop:Departure delay >= LATE_THRESHOLD_MIN` (default 5, configurable).
- **v1 exclusions:** drop cancelled departures, since cancellation is a different problem. Restrict to passenger services and keep all operators. Record these decisions in `docs/decisions.md`.
- **Leakage: never use as features** anything only known after the fact:
  - `Service:Maximum delay`
  - any cancellation flag
  - `Stop:Actual platform`
  - `Stop:Platform change`
  - delays at other stops of the same service
  - observed weather *after* the departure hour

## 5. Features

All feature logic lives in **one module**, `spoorcast/features/`. Training and serving both call the same functions; this is non-negotiable, because it avoids training/serving skew.

**Schedule/calendar:**
- station code (categorical)
- service type
- company
- local hour
- day of week
- month
- is_weekend
- Dutch public holiday flag (small hard-coded or library-based list)
- is_rush_hour
- minutes since first departure of that service (position in the trip)

**Weather (at the departure hour, nearest weather point):**
- the variables listed in §3.3

**Historical aggregates (point-in-time correct):**
- trailing 28-day late rate for the station
- trailing 28-day late rate for the station × hour bucket
- trailing 28-day late rate for the service type

These must be computed using **only data strictly before the departure's service date**. Write a dedicated test proving it (see §7).

At serving time, these aggregates are looked up from a precomputed snapshot through a `FeatureLookup` protocol (§6.3). This is deliberately a seam: Lucas will later replace it with a Databricks feature table.

## 6. Components to build

### 6.1 Data pipeline (`spoorcast/data/`)

`download_services(months)`, `download_stations()`, `download_weather(points, start, end)`:
- Pure I/O, cached.

`build_dataset(months) -> path`:
- Joins services, stations and weather.
- Applies the label and exclusions.
- Computes features.
- Writes Parquet partitioned by `year_month`.

Data validation on load:
- expected columns
- dtypes
- null rates within sane bounds
- delays non-negative or within a plausible range

Fail loudly with a clear message.

### 6.2 Training (`spoorcast/training/`)

**Time-based split only.** Train on months `[t-N, t-1]`, validate on month `t`. N defaults to 12 and is configurable. No random splits.

**Baseline model:** historical late rate by station × hour bucket. The LightGBM model must be compared against it.

**Metrics:**
- ROC AUC
- PR AUC
- Brier score
- log loss
- calibration table (deciles)

Report both the model and the baseline.

`train(config) -> TrainResult`:
- `TrainResult` is a dataclass holding:
  - fitted `SpoorcastModel`
  - metrics dict
  - params
  - feature list
  - training window
  - a small input example DataFrame
- It **returns** results and does not log them anywhere. Lucas will add experiment tracking around this function.

Save locally via `SpoorcastModel.save(dir)`:
- `model.joblib`
- `metadata.json`: feature list, threshold, training window, metrics, package version, git SHA if available

### 6.3 Model wrapper (`spoorcast/model.py`)

`SpoorcastModel` wraps preprocessing (categorical encoding, column ordering) **and** the estimator.

Interface:
- `predict_proba(df: pd.DataFrame) -> np.ndarray` takes **raw feature-input columns** (the API-level fields plus looked-up/derived features).
- `save(dir)`
- `load(dir)`
- `input_schema`
- `metadata`

Design it so that wrapping it later as an MLflow pyfunc is a ~10-line job. Do not do the wrapping yourself.

### 6.4 Serving API (`spoorcast/serving/`)

**Endpoints:**

`POST /v1/predictions`:
- Request:
  - `station_code`
  - `scheduled_departure` (RFC 3339)
  - `service_type`
  - `company`
  - optional `service_start_time`
- Batch variant: accept a list (max 500).
- Response per item:
  - `prediction_id` (UUID)
  - `probability_late`
  - `late` (bool, at a configurable decision threshold)
  - `model_version`
  - `features_used` (optional debug flag)

Health endpoints:
- `GET /health`: liveness.
- `GET /ready`: model loaded and feature snapshot loaded.
- `GET /v1/model`: model metadata.

**Pluggable dependencies** (FastAPI dependency injection; each is a `Protocol` with a simple local implementation):
- `ModelProvider`: local implementation loads `SpoorcastModel` from a directory path (env var `SPOORCAST_MODEL_DIR`). *Seam: Lucas will add a registry-backed provider or proxy to a Databricks serving endpoint.*
- `FeatureLookup`: local implementation reads the aggregates snapshot Parquet. *Seam: online feature table.*
- `WeatherProvider`: `OpenMeteoForecastProvider` for real use, `FixedWeatherProvider` for tests, `HistoricalWeatherProvider` (reads cached archive data) for replay.
- `InferenceLogger`: local implementation writes JSON lines (request, features, prediction, `prediction_id`, model version, timestamp) to stdout or a file. *Seam: Lucas will route this to a Delta inference table for monitoring.*

**Other requirements:**
- Structured JSON logging.
- Request-ID middleware.
- Sensible 4xx errors for unknown station codes or bad timestamps.
- Generated OpenAPI docs with examples.
- Configuration via env vars (pydantic-settings).

### 6.5 Replay tool (`spoorcast replay`)

This is the drift-demo driver. Given a month, say a stormy one, it:
1. Reads that month's real departures.
2. Sends them to the API in schedule order, with an optional `--speedup` and `--limit`.
3. Uses the historical weather provider.
4. Writes `replay_labels.jsonl` mapping `prediction_id` to the actual `late` outcome, so realised performance can be joined later.

It is a client only. It must not compute drift or metrics; that is Lucas's part.

### 6.6 CLI (`spoorcast` via Typer)

Commands:
- `download`
- `build-dataset`
- `train`
- `evaluate`
- `snapshot-features` (writes the aggregates snapshot for serving)
- `serve`
- `replay`

Also add a `Makefile` with the same targets plus `lint`, `test`, `docker-build`.

## 7. Tests (required)

Use a small committed fixture (a few hundred rows of real-shaped data across two stations and a few days) under `tests/fixtures/`. No network in tests; mock HTTP.

Tests must cover:
- **Leakage test:** for a given departure, every aggregate feature only uses rows with service date strictly earlier than that departure's service date.
- **Train/serve parity test:** the same raw input produces identical features via the training path and the serving path.
- **Schema/validation tests:** bad input data fails with a clear error.
- **Model round-trip test:** train on the fixture, save, load, and get the same predictions.
- **API tests** (TestClient with fake providers):
  - happy path
  - batch
  - unknown station → 422/404
  - `/ready` false before load
  - `prediction_id` is unique
  - `InferenceLogger` is called once per prediction
- **End-to-end smoke test:** fixture → build → train → save → API predict.

## 8. Out of scope: Lucas builds these himself

Do **not** implement, scaffold or stub any of the following beyond the seams described above:

- MLflow experiment tracking, model logging, pyfunc wrapping, Unity Catalog model registry, aliases (`champion`/`challenger`)
- Databricks jobs, workflows, notebooks, Databricks Asset Bundles (`databricks.yml`)
- CI/CD pipelines (`.github/workflows/`, Azure DevOps YAML)
- Databricks Model Serving endpoints, AI Gateway or inference tables
- Feature Store / feature tables / online tables
- Lakehouse Monitoring, drift metrics, alerting, retraining triggers, champion/challenger evaluation logic
- Infrastructure as code (Terraform, Bicep), Kubernetes manifests, Helm
- Secrets management beyond reading env vars

Mark each seam in code with a comment of the form `# MLOPS-SEAM: <what plugs in here>`, and list them all in `docs/mlops-seams.md` with:
- file and line
- the protocol/function signature
- what Lucas is expected to plug in

## 9. Repo layout (target)

```
spoorcast/
├── pyproject.toml
├── uv.lock
├── Makefile
├── Dockerfile
├── README.md            # what it is, quickstart, data attribution
├── docs/
│   ├── decisions.md     # label definition, exclusions, split strategy, etc.
│   └── mlops-seams.md   # every MLOPS-SEAM, with signatures
├── src/spoorcast/
│   ├── config.py
│   ├── cli.py
│   ├── data/            # sources.py, ingest.py, validate.py, weather.py
│   ├── features/        # build.py, aggregates.py, calendar.py
│   ├── training/        # train.py, evaluate.py, baseline.py
│   ├── model.py
│   └── serving/         # app.py, schemas.py, providers.py, logging.py
└── tests/
    ├── fixtures/
    └── ...
```

## 10. Definition of done

1. `make lint` and `make test` pass on a clean checkout with no network access.
2. `spoorcast download --months 2025-09..2026-08`, `build-dataset`, and `train --validate-month 2026-08` work end-to-end on a laptop (16 GB RAM). If memory is tight, sample or restrict operators, and document how.
3. The trained model beats the baseline on PR AUC, or `docs/decisions.md` explains why not.
4. `make serve` followed by a `curl` to `/v1/predictions` returns a prediction. `make docker-build` produces an image that runs the API with a mounted model directory.
5. `spoorcast replay --month 2026-02 --limit 1000` runs against the local API and writes `replay_labels.jsonl`.
6. `docs/mlops-seams.md` is complete and accurate.
7. The README credits Rijden de Treinen (CC BY 4.0) and the weather provider.

## 11. Working agreements

- Commit in small, logical steps with clear messages.
- If a requirement is ambiguous, pick the simplest reasonable option, record it in `docs/decisions.md`, and continue. Don't stall.
- Keep functions pure where possible, with I/O at the edges. That is what makes the seams work.
- Prefer clarity over cleverness. This code will be read by interviewers.