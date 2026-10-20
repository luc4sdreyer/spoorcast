"""docs/mlops-seams.md must list every MLOPS-SEAM comment at its current line."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = re.compile(r"`(src/spoorcast/[\w/]+\.py):(\d+)`")


def seam_comments() -> set[tuple[str, int]]:
    found = set()
    for path in (ROOT / "src").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if "# MLOPS-SEAM:" in line:
                found.add((str(path.relative_to(ROOT)), number))
    return found


def test_seams_doc_matches_code() -> None:
    documented = {
        (path, int(line))
        for path, line in REFERENCE.findall((ROOT / "docs" / "mlops-seams.md").read_text())
    }
    assert documented == seam_comments()
