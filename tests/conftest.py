"""Shared fixtures. Everything runs offline on the small extract in ``tests/fixtures``."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from spoorcast.config import Settings
from spoorcast.data.dataset import build_dataset, load_daily_stats
from spoorcast.data.ingest import load_stations
from spoorcast.features.snapshot import write_snapshot
from spoorcast.model import SpoorcastModel
from spoorcast.training.train import TrainConfig, TrainResult, train

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_MONTHS = ["2026-01", "2026-02"]
# LightGBM settings that make sense for a few hundred rows.
TINY_LGBM_PARAMS = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5}


def make_settings(root: Path) -> Settings:
    """Settings pointing at a fresh data directory seeded with the raw fixtures."""
    settings = Settings(data_dir=root / "data", model_dir=root / "model")
    settings.raw_dir.mkdir(parents=True)
    for name in ("services", "stations", "weather"):
        shutil.copytree(FIXTURES / name, settings.raw_dir / name)
    return settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture(scope="session")
def built_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings with the fixture dataset already built (shared, treat as read-only)."""
    settings = make_settings(tmp_path_factory.mktemp("built"))
    build_dataset(FIXTURE_MONTHS, settings)
    return settings


@pytest.fixture(scope="session")
def train_result(built_settings: Settings) -> TrainResult:
    return train(
        TrainConfig(
            validate_month="2026-02",
            train_months=1,
            dataset_dir=built_settings.dataset_dir,
            max_train_rows=None,
            lgbm_params=TINY_LGBM_PARAMS,
        )
    )


@pytest.fixture(scope="session")
def model_dir(train_result: TrainResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_result.model.save(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="session")
def saved_model(model_dir: Path) -> SpoorcastModel:
    return SpoorcastModel.load(model_dir)


def snapshot_as_of(settings: Settings, as_of: date, directory: Path) -> Path:
    return write_snapshot(
        directory,
        load_daily_stats(settings),
        load_stations(settings.stations_raw_path),
        as_of,
        settings.aggregate_window_days,
    )
