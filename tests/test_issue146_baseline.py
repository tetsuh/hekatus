"""Board-free preceding-record resolver tests for Issue #146 long planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import issue146_baseline as baseline
from tools.issue146_contract import AUTHORITATIVE_PLAN_CANONICAL_SHA256

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
BOARD_SERIAL = "0000040331914124"
BUS_ID = "0000:09:00.0"
RUN_ID = "issue146-long-run1-c018"
SNAPSHOT_NAME = f"pre-run-snapshot-{RUN_ID}.json"


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run1-c018"
    run_dir.mkdir()
    snapshot = {
        "schema": baseline.SNAPSHOT_SCHEMA,
        "stage": "pre_run",
        "status": "ok",
        "returncode": 0,
        "snapshot": {
            "device_info": [
                {
                    "board_info": {
                        "board_id": BOARD_SERIAL,
                        "serial": BOARD_SERIAL,
                        "bus_id": BUS_ID,
                    }
                }
            ]
        },
    }
    snapshot_path = run_dir / SNAPSHOT_NAME
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    snapshot_raw = snapshot_path.read_bytes()
    record = {
        "record_schema": baseline.RECORD_SCHEMA,
        "status": "ok",
        "plan_sha256": AUTHORITATIVE_PLAN_CANONICAL_SHA256,
        "run": {"mode": "long", "run_number": 1, "run_id": RUN_ID},
        "configuration": {"id": "C018"},
        "pre_run_snapshot": {
            "path": SNAPSHOT_NAME,
            "schema": baseline.SNAPSHOT_SCHEMA,
            "sha256": hashlib.sha256(snapshot_raw).hexdigest(),
            "byte_count": len(snapshot_raw),
            "selected_device": {"board_id": BOARD_SERIAL, "bus_id": BUS_ID},
        },
    }
    record_path = run_dir / f"run-{RUN_ID}.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, snapshot_path


def _resolve(record_path: Path, tmp_path: Path) -> str:
    return baseline.resolve_pre_snapshot(
        record_path,
        expected_run_number=1,
        expected_run_id=RUN_ID,
        expected_configuration="C018",
        plan_path=PLAN,
        repository_root=tmp_path,
    )


def test_issue146_baseline_resolver_returns_descriptor_path_only(tmp_path: Path) -> None:
    record_path, snapshot_path = _write_case(tmp_path)
    assert _resolve(record_path, tmp_path) == snapshot_path.relative_to(tmp_path).as_posix()


@pytest.mark.parametrize("mutation", ("bad_descriptor", "missing", "failed", "wrong_identity", "bad_hash"))
def test_issue146_baseline_resolver_rejects_invalid_predecessors(
    tmp_path: Path, mutation: str
) -> None:
    record_path, snapshot_path = _write_case(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if mutation == "bad_descriptor":
        record["pre_run_snapshot"]["path"] = "post-run-snapshot-other.json"
    elif mutation == "missing":
        record.pop("pre_run_snapshot")
    elif mutation == "failed":
        record["status"] = "failed"
    elif mutation == "wrong_identity":
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["snapshot"]["device_info"][0]["board_info"]["bus_id"] = "0000:0a:00.0"
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    elif mutation == "bad_hash":
        record["pre_run_snapshot"]["sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(baseline.BaselineResolutionError):
        _resolve(record_path, tmp_path)


def test_issue146_baseline_resolver_cli_fails_with_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = baseline.main(
        [
            "--plan",
            str(PLAN),
            "--record",
            str(tmp_path / "missing-record.json"),
            "--expected-run-number",
            "1",
            "--expected-run-id",
            RUN_ID,
            "--expected-configuration",
            "C018",
        ]
    )
    assert status != 0
    assert "baseline resolver rejected" in capsys.readouterr().err


def test_issue146_baseline_resolver_rejects_snapshot_run_identity_mismatch(
    tmp_path: Path,
) -> None:
    record_path, snapshot_path = _write_case(tmp_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["requested_run_id"] = "issue146-long-run-other-c018"
    snapshot_raw = json.dumps(snapshot).encode("utf-8")
    snapshot_path.write_bytes(snapshot_raw)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["pre_run_snapshot"]["sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
    record["pre_run_snapshot"]["byte_count"] = len(snapshot_raw)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(baseline.BaselineResolutionError, match="requested_run_id"):
        _resolve(record_path, tmp_path)
