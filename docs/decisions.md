# Decisions

Choices made while building Spoorcast, with the reasoning. Where the brief was ambiguous,
the simplest reasonable option was picked and recorded here.

## Prediction task

**Unit of prediction.** One scheduled departure of one service from one station: archive
rows where `Stop:Departure time` is set. Arrival-only rows (terminating stops) are dropped.

**Label.** `late = Stop:Departure delay >= 5` minutes. The threshold is
`SPOORCAST_LATE_THRESHOLD_MIN`, applied when the dataset is built. The value used is stored in
`data/processed/dataset/_dataset.json` and copied into the model's `metadata.json`, so a model
always records which label it was trained on. The aggregates are late rates under the same
threshold, so changing it means rebuilding the dataset.

**Exclusions (v1).** In this order, with counts logged per month:

| Excluded | Why | Typical rows / month |
|---|---|---|
| Cancelled departures (`Stop:Departure cancelled`) | Cancellation is a different problem | 60k–220k |
| Service types containing "bus" (`Stopbus ipv trein`, `Snelbus ipv trein`, `Taxibus ipv trein`, `Bus`) | Rail-replacement buses are not trains | 25k–65k |
| Stations outside the Netherlands (station list `country != NL`) | The task is about Dutch stations; the weather points cover only NL | 80k–100k |
| Stations missing from the station list | No coordinates, so no weather; ~0.15% (codes added after the 2023-09 list) | ~2.5k |
| Delay above 720 minutes | Data artefacts (24h-shifted records with delay ≈ 1440) | 0–60 |

All operators are kept (NS, Arriva, Keolis, international operators like NS Int, DB or NMBS),
as long as the departure is from a Dutch station by train. That makes `company` and
`service_type` features with many levels, which LightGBM handles natively.

**Station list.** Rijden de Treinen publishes snapshots; the newest is `stations-2023-09.csv`.
It is used for every month.

## Data

**Monthly files.** The monthly archive (`services-YYYY-MM.csv.gz`) is used rather than the
yearly one, because it mirrors how new data arrives. Downloads go to `data/raw/`, are skipped
when present, resume from a `.part` file (HTTP `Range`), and fall back to the DuckDB mirror.

**Validation.** Raw files must contain the required columns; the optional ones (platform
columns, cancellation flags, maximum delay) may be missing because none of them is used.
Parsing uses explicit DuckDB types, so a malformed value fails the load with the file name.
After loading: null-rate bounds per column, tz-aware timestamps, no negative delays, and at
most 0.1% of departures above 720 minutes. Station and weather tables have their own checks
(duplicate codes, coordinate ranges, physical ranges for each weather variable). The dataset
build fails if more than 5% of a month's departures have no weather (usually: the weather was
not downloaded).

**Service date and start time.** `Service:Date` is used as the departure's date for the
point-in-time aggregates (it can differ from the calendar date after midnight).
`service_start_time` is the service's first scheduled departure in the raw file, computed
before any filtering.

**Time zones.** All timestamps are stored in UTC. Calendar features convert to
`Europe/Amsterdam`, so they follow DST (tested on the March switch).

## Features

All feature logic is in `src/spoorcast/features/`. `build_features(inputs, weather,
aggregates)` is the single assembly function. The training path
(`features.build.training_features`) and the serving path (`serving.predict.serving_features`)
differ only in where they look up weather and aggregates.

**Calendar.** Local hour, day of week, month, weekend, holiday, rush hour. Holidays are
computed in code (Easter-based dates, King's Day moved to Saturday when it falls on a Sunday,
Liberation Day every year) to avoid a dependency. Rush hour is the NS peak: weekdays,
not holidays, 06:30–09:00 and 16:00–18:30.

With a 12-month window each `month` value is seen in one year only, so `month` mostly acts as
a seasonal proxy. When validating on a month whose previous-year counterpart is not in the
window, the model has never seen that month value. For 2026-08 this happens because 2025-08
is not downloaded (see below).

**Trip position.** `minutes_since_service_start` = scheduled departure − service start. The
API field `service_start_time` is optional. Without it the feature is missing; LightGBM treats
missing values that never occurred in training as zero, which means "first stop". This is a
known limitation; clients should send the start time.

**Weather.** 21 representative points cover the Netherlands; each station is mapped to the
nearest (haversine). Hourly Open-Meteo data per point and month are cached as JSON. A
departure gets the record at the start of its hour (UTC floor). Open-Meteo's hourly
precipitation and snowfall are sums over the preceding hour, and temperature and wind are
instantaneous values at that timestamp, so nothing observed after the departure is used.
Training uses the Historical Weather API (observed/reanalysis). Serving uses the Forecast API,
cached per point for 15 minutes; if it is unavailable, the weather features are missing and
the prediction still succeeds. Training on observed weather and serving on forecasts is a
known train/serve difference that the parity test cannot cover; replays use the archive, which
removes it.

**Historical aggregates.** Late rate over the 28 days *strictly before* the service date
(`[D-28, D-1]`), for the station, for the station × hour bucket (night 0–5, am_peak 6–8,
midday 9–15, pm_peak 16–18, evening 19–23, local time), and for the service type. Rates are
computed from per-day counts (`data/processed/daily_stats`) with one SQL range join
(`features.aggregates.trailing_rates`). Training calls it once per service date; the serving
snapshot calls it once for the snapshot date. Keys with no departures in the window get a
missing value.

The build uses the daily stats of every month already on disk. That makes incremental builds
match full rebuilds, but the first built month (2025-09 here) has partial history in its
first four weeks. A key with no history gets a missing value, which the model can handle.

**Serving snapshot.** `spoorcast snapshot-features` writes the aggregates as of one date
(default: the day after the last data day) plus the Dutch station directory. The API looks
features up by key, regardless of the request date. For a replay of month M the snapshot
should be as of M-01 (`make serve-replay` does this).

## Training

**Split.** Time-based only: train on `[t-N, t-1]` (N = 12), validate on `t`. Missing months
in the window are skipped with a warning. With the suggested download (2025-09..2026-08) and
`--validate-month 2026-08`, the window is 2025-08..2026-07 and 2025-08 is missing, so
training uses 11 months. To use the full 12, also download and build 2025-08.

**Memory.** The target is a 16 GB laptop; development used 7 GB. The dataset is built one
month at a time (peak ≈ 1.5 GB). The training window has about 16M rows; training uses a
uniform random sample of 3M rows (`--max-train-rows`, `0` disables sampling), drawn per month
with a fixed seed. Peak memory for training is about 2.5 GB. The validation month is not
sampled.

**Model.** LightGBM binary classifier with fixed parameters (400 trees, learning rate 0.05,
63 leaves, at least 200 rows per leaf, row and column subsampling). Categorical columns use
LightGBM's native categorical handling; category values are fixed at training, so unseen
values at serving become missing. No early stopping: the only held-out data is the validation
month, and stopping on it would bias the reported metrics. No class weighting, so the
probabilities stay calibrated.

**Baseline.** The late rate per station × hour bucket over the training window, falling back
to the station rate and then the global rate when a cell has fewer than 20 departures.

**Metrics.** ROC AUC, PR AUC (average precision), Brier score, log loss and a calibration table
by decile of predicted probability, for both model and baseline. `train()` returns them in
`TrainResult` and stores the scalars in `metadata.json`.

**Decision threshold.** The API's `late` flag is `probability_late >= 0.2`
(`SPOORCAST_DECISION_THRESHOLD`). With a base rate around 4%, the conventional 0.5 would almost
never fire. 0.2 picks out the high-risk tail. The right value depends on the use case.

## Results

Validation on 2026-08 (1,461,550 departures, 3.78% late). Trained on 2025-09..2026-07
(3M sampled rows).

| | ROC AUC | PR AUC | Brier | Log loss |
|---|---|---|---|---|
| LightGBM | 0.717 | 0.138 | 0.0346 | 0.148 |
| Baseline | 0.651 | 0.061 | 0.0362 | 0.157 |

The model beats the baseline on PR AUC by a factor of about 2.3. Its top decile has an
observed late rate of 11.8%, about 3× the base rate. The model slightly underestimates in
August (mean prediction below the observed rate in every decile), which is consistent with
the training window's seasonal mix. The baseline overestimates.

Calibration, model (deciles of predicted probability):

| decile | mean predicted | observed |
|---|---|---|
| 1 | 0.004 | 0.008 |
| 2 | 0.008 | 0.013 |
| 3 | 0.012 | 0.017 |
| 4 | 0.017 | 0.021 |
| 5 | 0.022 | 0.026 |
| 6 | 0.027 | 0.031 |
| 7 | 0.033 | 0.037 |
| 8 | 0.042 | 0.047 |
| 9 | 0.055 | 0.062 |
| 10 | 0.110 | 0.118 |

Most important features (gain): station, station × hour late rate, minutes since service
start, operator, service-type late rate, station late rate, temperature.

## API

* One endpoint per shape: `POST /v1/predictions` (one object) and
  `POST /v1/predictions/batch` (`{"instances": [...]}`, 1–500). This keeps request and
  response schemas unambiguous in OpenAPI.
* Unknown station codes return **422** with the list of unknown codes (the whole batch is
  rejected). A station is known if it is in the snapshot's Dutch station directory. Codes are
  upper-cased.
* Timestamps must include a UTC offset (naive timestamps return 422);
  `service_start_time` after `scheduled_departure` returns 422.
* If the model or snapshot fails to load, the process still starts. `/ready` returns 503 and
  prediction endpoints return 503.
* `features_used` is only returned with `?debug=true`; the inference log always contains the
  features.
