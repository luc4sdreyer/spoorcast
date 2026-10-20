"""Downloads of the raw source files. Pure I/O: every function is cached and resumable.

A file that already exists at its destination is never downloaded again. Partial downloads
are kept as ``<name>.part`` and resumed with an HTTP ``Range`` request on the next attempt.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import httpx

from spoorcast.config import SERVICES_URL_TEMPLATES, STATIONS_URL, Settings

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20


class DownloadError(RuntimeError):
    """Raised when a file could not be fetched from any of its URLs."""


def default_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(60.0, connect=15.0),
        follow_redirects=True,
        headers={"User-Agent": "spoorcast/0.1 (portfolio project)"},
    )


@contextmanager
def client_scope(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Use the caller's client as-is, or open (and close) a default one."""
    if client is not None:
        yield client
    else:
        with default_client() as owned:
            yield owned


def download_file(client: httpx.Client, urls: Sequence[str], dest: Path) -> Path:
    """Download the first URL that works to ``dest``, resuming a previous partial download."""
    if dest.exists():
        logger.debug("Cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    errors: list[str] = []
    for url in urls:
        try:
            _stream_to(client, url, partial)
        except httpx.HTTPError as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("Download failed from %s: %s", url, exc)
            continue
        partial.rename(dest)
        logger.info("Downloaded %s", dest)
        return dest
    raise DownloadError(f"Could not download {dest.name}:\n  " + "\n  ".join(errors))


def _stream_to(client: httpx.Client, url: str, partial: Path) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == 416:  # the partial file is already complete
            return
        response.raise_for_status()
        # A 200 (instead of 206) means the server ignored the Range header: start over.
        mode = "ab" if response.status_code == 206 else "wb"
        with partial.open(mode) as fh:
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                fh.write(chunk)


def services_path(settings: Settings, month: str) -> Path:
    return settings.services_raw_dir / f"services-{month}.csv.gz"


def download_services(
    months: Iterable[str], settings: Settings, client: httpx.Client | None = None
) -> list[Path]:
    """Download the monthly Rijden de Treinen service archives (one row per stop)."""
    with client_scope(client) as http:
        return [
            download_file(
                http,
                [template.format(month=month) for template in SERVICES_URL_TEMPLATES],
                services_path(settings, month),
            )
            for month in months
        ]


def download_stations(settings: Settings, client: httpx.Client | None = None) -> Path:
    """Download the Rijden de Treinen station list (code, name, country, coordinates)."""
    with client_scope(client) as http:
        return download_file(http, [STATIONS_URL], settings.stations_raw_path)
