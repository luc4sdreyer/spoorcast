# MLOps seams

Spoorcast's application code stops at clean interfaces. Everything listed here is where the
platform side (experiment tracking, registry, feature tables, serving, inference tables,
monitoring, jobs) plugs in. Each seam is marked in the code with a `# MLOPS-SEAM:` comment;
`tests/test_docs.py` checks that this file and those comments stay in sync.

| # | Location | What plugs in |
|---|---|---|
| 1 | `src/spoorcast/training/train.py:128` | Experiment tracking around `train()` |
| 2 | `src/spoorcast/model.py:36` | pyfunc wrapper + model logging |
| 3 | `src/spoorcast/cli.py:98` | Model registration after training |
| 4 | `src/spoorcast/serving/providers.py:57` | Registry-backed or remote `ModelProvider` |
| 5 | `src/spoorcast/serving/providers.py:103` | Online feature table `FeatureLookup` |
| 6 | `src/spoorcast/features/snapshot.py:42` | Feature table writer (offline + online) |
| 7 | `src/spoorcast/serving/providers.py:276` | Inference table `InferenceLogger` |
| 8 | `src/spoorcast/serving/app.py:89` | Injection point for all serving providers |
| 9 | `src/spoorcast/data/dataset.py:125` | Scheduled data/feature job per arriving month |
| 10 | `src/spoorcast/replay.py:81` | Ground-truth labels for monitoring joins |

## 1. Experiment tracking: `train()`

`src/spoorcast/training/train.py:128`

```python
def train(config: TrainConfig) -> TrainResult: ...

@dataclass
class TrainResult:
    model: SpoorcastModel
    baseline: StationHourBaseline
    metrics: dict[str, float]           # model_roc_auc, model_pr_auc, model_brier, model_log_loss,
                                        # baseline_* (same), model_base_rate, n_train, ...
    calibration: dict[str, pd.DataFrame]  # "model" / "baseline" decile tables
    params: dict[str, Any]              # full LightGBM parameters
    feature_columns: list[str]
    training_window: dict[str, Any]     # {"train_months": [...], "validate_month": "YYYY-MM"}
    input_example: pd.DataFrame         # 5 validation rows of model input
```

`train()` has no side effects: it returns everything and logs nothing. Plug in: start a run,
call `train()`, log `params`, `metrics`, the calibration tables (as artifacts), the training
window (as tags) and the model (with `input_example` for its signature).

## 2. Model packaging: `SpoorcastModel`

`src/spoorcast/model.py:36`

```python
class SpoorcastModel:
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray   # 1-D P(late)
    def save(self, directory: Path | str) -> Path             # model.joblib + metadata.json
    @classmethod
    def load(cls, directory: Path | str) -> SpoorcastModel
    input_schema: list[dict[str, str]]                        # [{"name", "type"}], string|double
    metadata: dict[str, Any]
```

The model takes the feature frame from `spoorcast.features.build.build_features` and does its
own column ordering and categorical encoding, so the pyfunc wrapper is just `load_context`
(call `SpoorcastModel.load` on the logged directory) and `predict` (call `predict_proba`). The
module docstring has the ~10-line sketch. `input_schema` maps onto a column-based signature.

## 3. Model registration: `spoorcast train`

`src/spoorcast/cli.py:98`

The CLI saves the trained model to `--output` (default `models/latest`). Plug in: register the
logged model and set an alias. Champion/challenger logic can start from
`spoorcast.training.train.evaluate_saved_model(model_dir, month, dataset_dir)`, which scores a
saved model (and the baseline) on any month.

## 4. `ModelProvider`

`src/spoorcast/serving/providers.py:57`

```python
class ModelProvider(Protocol):
    def load(self) -> None: ...          # called once at startup (lifespan)
    def is_ready(self) -> bool: ...      # drives /ready
    def get(self) -> LoadedModel: ...    # LoadedModel(model: SpoorcastModel, version: str)
```

Local implementation: `LocalModelProvider(model_dir)` (env `SPOORCAST_MODEL_DIR`). Plug in:
a provider that loads the model behind a registry alias (`version` = registry version), or
one whose `model.predict_proba` proxies to a managed serving endpoint. The API only uses
`version` and `model.predict_proba` / `model.metadata`.

## 5. `FeatureLookup`

`src/spoorcast/serving/providers.py:103`

```python
class FeatureLookup(Protocol):
    def load(self) -> None: ...
    def is_ready(self) -> bool: ...
    def station(self, station_code: str) -> StationInfo | None: ...  # None -> HTTP 422
    def aggregates(self, inputs: pd.DataFrame) -> pd.DataFrame: ...
    def info(self) -> dict[str, Any]: ...                            # shown in /v1/model
```

`aggregates()` receives the canonical inputs (`station_code`, `service_type`, `company`,
`scheduled_departure`, `service_start_time`) and must return the columns
`station_late_rate_28d`, `station_hour_late_rate_28d`, `service_type_late_rate_28d`, aligned
with the input index. The keys are `station_code`, `(station_code, hour_bucket)` and
`service_type`; `hour_bucket` comes from `spoorcast.features.calendar.hour_bucket`.

Local implementation: `SnapshotFeatureLookup(snapshot_dir)` (env `SPOORCAST_SNAPSHOT_DIR`).
Plug in: online-table lookups on those keys.

## 6. Feature table writer: `write_snapshot()`

`src/spoorcast/features/snapshot.py:42`

```python
def write_snapshot(directory: Path, daily: DailyCounts, stations: pd.DataFrame,
                   as_of: date, window_days: int) -> Path
```

The values come from `spoorcast.features.aggregates.build_snapshot(daily, as_of,
window_days)`, which calls the same `trailing_rates()` used to build training features. Plug
in: write the per-day `daily_stats` (or the trailing rates per as-of date) to a feature table
keyed by the aggregate keys plus a timestamp. That gives point-in-time lookups for training and
an online table for serving, replacing both the snapshot and the Parquet join.

## 7. `InferenceLogger`

`src/spoorcast/serving/providers.py:276`

```python
class InferenceLogger(Protocol):
    def log(self, record: dict[str, Any]) -> None: ...
```

Called exactly once per prediction with:

```json
{"prediction_id": "uuid", "request_id": "...", "timestamp": "ISO-8601 UTC",
 "model_version": "...", "request": {...API fields...}, "features": {...model input...},
 "probability_late": 0.07, "late": false, "decision_threshold": 0.2}
```

Local implementation: `JsonLinesInferenceLogger(path)` (env `SPOORCAST_INFERENCE_LOG_PATH`,
stdout when unset). Plug in: append to an inference table (directly, or via a queue).

## 8. Provider injection: `create_app()`

`src/spoorcast/serving/app.py:89`

```python
def create_app(settings: Settings | None = None, *,
               model_provider: ModelProvider | None = None,
               feature_lookup: FeatureLookup | None = None,
               weather_provider: WeatherProvider | None = None,
               inference_logger: InferenceLogger | None = None) -> FastAPI
```

Any argument left out gets the local implementation chosen from settings. Platform
implementations are passed here (or chosen in `create_app` from a new setting).

## 9. Data/feature job: `build_dataset()`

`src/spoorcast/data/dataset.py:125`

```python
def build_dataset(months: Sequence[str], settings: Settings) -> Path
```

Idempotent per month and incremental: it uses the daily stats of every month already built,
so running it for one newly arrived month gives the same result as a full rebuild. Plug in:
schedule it per month (after `download_services` / `download_weather`) and write the
`departures`, `daily_stats` and `dataset` partitions to tables partitioned by `year_month`.
Retraining triggers can watch for a new partition.

## 10. Ground truth for monitoring: `replay()`

`src/spoorcast/replay.py:81`

```python
def replay(departures: pd.DataFrame, client: httpx.Client, labels_path: Path, *,
           batch_size: int = 100, speedup: float | None = None, ...) -> ReplayResult
```

Writes `replay_labels.jsonl`, one line per prediction:

```json
{"prediction_id": "uuid", "late": true, "departure_delay_min": 7, "service_id": 17654865,
 "station_code": "GD", "scheduled_departure": "2026-02-01T05:30:00+00:00"}
```

Join on `prediction_id` with the inference log (seam 7) to get realised performance; drift
metrics compare the logged `features` against the training data. The replay itself computes
nothing.
