import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe_path() -> Path:
    return Path(__file__).parent / "probe_reexport.py"


def test_no_reexport_errors(probe_path: Path) -> None:
    if shutil.which("mypy") is None:
        pytest.skip("mypy not on PATH")

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(probe_path)],
        capture_output=True,
        text=True,
    )

    reexport_errors = [
        line
        for line in result.stdout.splitlines()
        if "does not explicitly export attribute" in line
    ]
    assert reexport_errors == [], (
        "mypy --strict found re-export errors:\n" + "\n".join(reexport_errors)
    )
