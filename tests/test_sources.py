"""Downloads are cached, resumable and fall back to the mirror. HTTP is mocked."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from conftest import FIXTURES
from spoorcast.config import Settings
from spoorcast.data.sources import DownloadError, download_file, download_services
from spoorcast.data.weather import WEATHER_POINTS_BY_ID, download_weather, load_weather_table

PAYLOAD = b"x" * 1000


def client_for(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_download_is_cached(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=PAYLOAD)

    dest = tmp_path / "file.bin"
    with client_for(httpx.MockTransport(handler)) as client:
        download_file(client, ["https://a/file.bin"], dest)
        download_file(client, ["https://a/file.bin"], dest)
    assert dest.read_bytes() == PAYLOAD
    assert len(calls) == 1


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:400])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=400-"
        return httpx.Response(206, content=PAYLOAD[400:])

    with client_for(httpx.MockTransport(handler)) as client:
        download_file(client, ["https://a/file.bin"], dest)
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.bin.part").exists()


def test_download_restarts_when_range_is_ignored(tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(b"garbage")
    with client_for(httpx.MockTransport(lambda r: httpx.Response(200, content=PAYLOAD))) as c:
        download_file(c, ["https://a/file.bin"], dest)
    assert dest.read_bytes() == PAYLOAD


def test_download_falls_back_to_mirror(tmp_settings_root: Path) -> None:
    settings = Settings(data_dir=tmp_settings_root)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "opendata.rijdendetreinen.nl":
            return httpx.Response(503)
        return httpx.Response(200, content=PAYLOAD)

    with client_for(httpx.MockTransport(handler)) as client:
        [path] = download_services(["2026-02"], settings, client)
    assert path.name == "services-2026-02.csv.gz"
    assert path.read_bytes() == PAYLOAD


def test_download_error_lists_every_url(tmp_path: Path) -> None:
    with (
        client_for(httpx.MockTransport(lambda r: httpx.Response(404))) as client,
        pytest.raises(DownloadError, match=r"https://a/f(.|\n)*https://b/f"),
    ):
        download_file(client, ["https://a/f", "https://b/f"], tmp_path / "f")


@pytest.fixture
def tmp_settings_root(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_weather_download_caches_final_months_only(tmp_settings_root: Path) -> None:
    settings = Settings(data_dir=tmp_settings_root)
    payload = json.loads((FIXTURES / "weather" / "utrecht" / "2026-02.json").read_text())
    requested: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(dict(request.url.params))
        return httpx.Response(200, json=payload)

    point = WEATHER_POINTS_BY_ID["utrecht"]
    with client_for(httpx.MockTransport(handler)) as client:
        for _ in range(2):
            download_weather(
                [point],
                date(2026, 1, 1),
                date(2026, 2, 28),
                settings,
                client,
                today=date(2026, 2, 20),
            )
    # January is final and fetched once; February is still in progress and refetched.
    months = [(r["start_date"], r["end_date"]) for r in requested]
    assert months.count(("2026-01-01", "2026-01-31")) == 1
    assert months.count(("2026-02-01", "2026-02-19")) == 2
    assert requested[0]["timezone"] == "GMT"
    table = load_weather_table(settings.weather_raw_dir)
    assert set(table["weather_point_id"]) == {"utrecht"}
