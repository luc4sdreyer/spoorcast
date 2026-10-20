"""End to end: fixture -> build -> train -> save -> snapshot -> API predict -> replay."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from conftest import TINY_LGBM_PARAMS
from spoorcast.cli import app as cli
from spoorcast.config import Settings
from spoorcast.replay import load_replay_departures, replay
from spoorcast.serving.app import create_app
from spoorcast.training.train import TrainConfig, train


def test_pipeline_to_api_and_replay(settings: Settings, tmp_path: Path) -> None:
    runner = CliRunner()
    env = {"SPOORCAST_DATA_DIR": str(settings.data_dir)}
    result = runner.invoke(cli, ["build-dataset", "--months", "2026-01..2026-02"], env=env)
    assert result.exit_code == 0, result.output

    training = train(
        TrainConfig(
            validate_month="2026-02",
            train_months=1,
            dataset_dir=settings.dataset_dir,
            lgbm_params=TINY_LGBM_PARAMS,
        )
    )
    training.model.save(settings.model_dir)

    result = runner.invoke(cli, ["snapshot-features"], env=env)
    assert result.exit_code == 0, result.output
    assert "as of 2026-02-04" in result.output

    log_path = tmp_path / "inference.jsonl"
    serving_settings = settings.model_copy(
        update={"weather_provider": "historical", "inference_log_path": log_path}
    )
    with TestClient(create_app(serving_settings)) as client:
        assert client.get("/ready").json()["ready"] is True
        response = client.post(
            "/v1/predictions",
            json={
                "station_code": "WD",
                "scheduled_departure": "2026-02-03T17:20:00+01:00",
                "service_type": "Intercity",
                "company": "NS",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 < body["probability_late"] < 1.0
        assert body["model_version"] == training.model.metadata["model_version"]

        departures = load_replay_departures(settings, "2026-02", limit=40)
        labels_path = tmp_path / "replay_labels.jsonl"
        outcome = replay(departures, client, labels_path, batch_size=16)

    assert outcome.sent == 40 and outcome.weather_provider == "historical"
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert len(labels) == 40
    assert {"prediction_id", "late", "departure_delay_min"} <= set(labels[0])
    assert [lab["scheduled_departure"] for lab in labels] == sorted(
        lab["scheduled_departure"] for lab in labels
    )

    # Every replayed prediction is in the inference log, so labels can be joined later.
    logged = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(logged) == 41
    assert {lab["prediction_id"] for lab in labels} <= {rec["prediction_id"] for rec in logged}


def test_replay_speedup_waits_in_schedule_time(built_settings: Settings, tmp_path: Path) -> None:
    departures = load_replay_departures(built_settings, "2026-02", limit=6)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/model":
            return httpx.Response(200, json={"weather_provider": "historical"})
        n = len(json.loads(request.content)["instances"])
        return httpx.Response(
            200, json={"predictions": [{"prediction_id": f"id-{i}"} for i in range(n)]}
        )

    waits: list[float] = []  # recorded instead of slept
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api") as client:
        replay(
            departures,
            client,
            tmp_path / "labels.jsonl",
            batch_size=2,
            speedup=60.0,
            sleep=waits.append,
        )
    # At 60x, a batch whose first departure is planned m minutes after the first is sent
    # about m seconds after the start.
    first = departures["scheduled_departure"].iloc[0]
    offsets = [
        (departures["scheduled_departure"].iloc[i] - first).total_seconds() / 60.0
        for i in range(0, 6, 2)
    ]
    assert waits == pytest.approx([o for o in offsets if o > 0], abs=0.5)
    assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 6
