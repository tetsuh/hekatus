"""Real P admission/session/launcher/publication using only isolated CPU children."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _fixture_repository(repository, destination):
    # Keep the real repository-relative fixture boundary, without changing any
    # measured source or writing its out/ tree. Copies cannot mutate originals.
    for relative in (
        "enodia", "tools", "docs/measurements/issue120-production",
        "docs/measurements/issue120-production-c2-prep",
        "docs/measurements/issue120-production-p-prep",
    ):
        shutil.copytree(repository / relative, destination / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in (repository / "docs/measurements").glob("*issue120*.json"):
        shutil.copy2(path, destination / "docs/measurements" / path.name)


@pytest.mark.slow
def test_p_pair_cpu_e2e_and_fail_closed_boundaries(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    original_output = repository / "out/issue120-production-p-prep"
    before = set(original_output.iterdir()) if original_output.exists() else set()
    _fixture_repository(repository, tmp_path)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1",
        GIT_DIR=subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=repository, text=True).strip(),
        GIT_WORK_TREE=str(tmp_path),
    )
    completed = subprocess.run(
        [sys.executable, "-c", ("import json; from tools import issue120_production_dry_run as d; "
                                "print(json.dumps(d.run()))")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert (tmp_path / result["fixture_root"] / "dry-run.json").is_file()
    assert (set(original_output.iterdir()) if original_output.exists() else set()) == before
    assert result["fixture_only"] is True
    assert result["P_executed"] is False
    assert result["native_invoked"] is False
    assert result["positive_launches"] == 2
    assert result["P1_exit"] == result["P2_exit"] == 0
    assert result["comparison_recomputed"] is True
    for key in ("boot", "head", "missing_receipt", "modified_receipt", "missing_P1", "failed_P1", "missing_approval", "dirty_source", "environment_summary", "historical_P1", "missing_adoption", "mismatched_adoption", "used_roles", "unacknowledged_P1"):
        assert result["rejected"][key] is True
