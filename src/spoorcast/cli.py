"""The ``spoorcast`` command line."""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from spoorcast.config import Settings
from spoorcast.months import month_bounds, parse_months

app = typer.Typer(help="Spoorcast: will this Dutch train depart late?", no_args_is_help=True)

MonthsOpt = Annotated[str, typer.Option(help="Months: 2025-09..2026-08, 2026-01,2026-02, ...")]
ModelDirOpt = Annotated[Path | None, typer.Option(help="Model directory [SPOORCAST_MODEL_DIR]")]


def _setup(verbose: bool = False) -> Settings:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return Settings()


def _print_metrics(metrics: dict[str, float]) -> None:
    names = sorted({k.split("_", 1)[1] for k in metrics if k.startswith(("model_", "baseline_"))})
    rows = {
        prefix: [metrics.get(f"{prefix}_{name}", math.nan) for name in names]
        for prefix in ("model", "baseline")
    }
    table = pd.DataFrame(rows, index=names)
    typer.echo(table.to_string(float_format=lambda v: f"{v:.4f}"))


@app.command()
def download(
    months: MonthsOpt,
    weather: Annotated[bool, typer.Option(help="Also download weather")] = True,
) -> None:
    """Download train services, stations and weather (cached; safe to re-run)."""
    from spoorcast.data.sources import download_services, download_stations
    from spoorcast.data.weather import WEATHER_POINTS, download_weather

    settings = _setup()
    month_list = parse_months(months)
    download_stations(settings)
    download_services(month_list, settings)
    if weather:
        start, _ = month_bounds(month_list[0])
        _, end = month_bounds(month_list[-1])
        download_weather(WEATHER_POINTS, start, end, settings)
    typer.echo(f"Downloaded {len(month_list)} month(s) into {settings.raw_dir}")


@app.command("build-dataset")
def build_dataset_cmd(months: MonthsOpt) -> None:
    """Clean, label and featurise months into data/processed/dataset."""
    from spoorcast.data.dataset import build_dataset

    settings = _setup()
    path = build_dataset(parse_months(months), settings)
    typer.echo(f"Dataset written to {path}")


@app.command()
def train(
    validate_month: Annotated[str, typer.Option(help="Month t to validate on (YYYY-MM)")],
    train_months: Annotated[int, typer.Option(help="Train on months [t-N, t-1]")] = 12,
    max_train_rows: Annotated[
        int, typer.Option(help="Sample the training window to this many rows (0: all)")
    ] = 3_000_000,
    output: ModelDirOpt = None,
    seed: int = 42,
) -> None:
    """Train the LightGBM model, compare it with the baseline and save it."""
    from spoorcast.training.train import TrainConfig
    from spoorcast.training.train import train as run_training

    settings = _setup()
    result = run_training(
        TrainConfig(
            validate_month=validate_month,
            train_months=train_months,
            dataset_dir=settings.dataset_dir,
            max_train_rows=max_train_rows or None,
            seed=seed,
        )
    )
    _print_metrics(result.metrics)
    for name, table in result.calibration.items():
        typer.echo(f"\nCalibration ({name}):\n{table.round(4).to_string(index=False)}")
    # MLOPS-SEAM: log/register the model here instead of (or as well as) saving locally.
    target = result.model.save(output or settings.model_dir)
    typer.echo(f"\nModel {result.model.metadata['model_version']} saved to {target}")


@app.command()
def evaluate(
    month: Annotated[str, typer.Option(help="Month to evaluate on (YYYY-MM)")],
    model_dir: ModelDirOpt = None,
) -> None:
    """Score a saved model (and the baseline) on a month of the dataset."""
    from spoorcast.training.train import evaluate_saved_model

    settings = _setup()
    metrics, calibration = evaluate_saved_model(
        model_dir or settings.model_dir, month, settings.dataset_dir
    )
    _print_metrics(metrics)
    for name, table in calibration.items():
        typer.echo(f"\nCalibration ({name}):\n{table.round(4).to_string(index=False)}")


@app.command("snapshot-features")
def snapshot_features(
    as_of: Annotated[
        str | None,
        typer.Option(help="Snapshot date (YYYY-MM-DD); default: day after the last data day"),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Snapshot directory")] = None,
) -> None:
    """Write the aggregates snapshot used by the API's FeatureLookup."""
    from spoorcast.data.dataset import load_daily_stats
    from spoorcast.data.ingest import load_stations
    from spoorcast.features.snapshot import write_snapshot

    settings = _setup()
    daily = load_daily_stats(settings)
    if as_of:
        snapshot_date = date.fromisoformat(as_of)
    else:
        last = max(frame["service_date"].max() for frame in daily.values())
        snapshot_date = pd.Timestamp(last).date() + timedelta(days=1)
    target = write_snapshot(
        output or settings.resolved_snapshot_dir,
        daily,
        load_stations(settings.stations_raw_path),
        snapshot_date,
        settings.aggregate_window_days,
    )
    typer.echo(f"Feature snapshot as of {snapshot_date} written to {target}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
) -> None:
    """Run the prediction API (configure with SPOORCAST_* environment variables)."""
    import uvicorn

    uvicorn.run(
        "spoorcast.serving.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_config=None,
    )


@app.command("replay")
def replay_cmd(
    month: Annotated[str, typer.Option(help="Month of real departures to replay (YYYY-MM)")],
    api_url: Annotated[str, typer.Option(help="Base URL of the API")] = "http://127.0.0.1:8000",
    limit: Annotated[int | None, typer.Option(help="Only the first N departures")] = None,
    speedup: Annotated[
        float | None, typer.Option(help="Replay in real time, this many times faster")
    ] = None,
    batch_size: Annotated[int, typer.Option(min=1, max=500)] = 100,
    output: Annotated[Path, typer.Option(help="Labels file")] = Path("replay_labels.jsonl"),
) -> None:
    """Send a month of departures to the API in schedule order; write replay_labels.jsonl."""
    import httpx

    from spoorcast.replay import load_replay_departures, replay

    settings = _setup()
    departures = load_replay_departures(settings, month, limit)
    with httpx.Client(base_url=api_url, timeout=60.0) as client:
        result = replay(departures, client, output, batch_size=batch_size, speedup=speedup)
    typer.echo(f"Replayed {result.sent} departures; labels written to {result.labels_path}")


if __name__ == "__main__":
    app()
