"""Non-record JSON companions stay visible without becoming timing evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import audit_issue122_aiclk as audit


def write_trace(path):
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,40,800,50\n"
        "2026-01-01T00:00:02+00:00,70,1350,51\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("value", "kind"), [
    ([], "array"),
    ([{"status": "complete", "clock": {"aiclk_mhz_for_elapsed_seconds": 1350}, "run_start": "2026-01-01T00:00:00Z", "run_end": "2026-01-01T00:00:02Z"}], "array"),
    (None, "null"),
    ("opaque payload | not a record", "string"),
    (7, "number"),
    (3.25, "number"),
    (True, "boolean"),
    (False, "boolean"),
])
def test_non_mapping_roots_are_typed_diagnostics_not_measurements(tmp_path, monkeypatch, value, kind):
    path = tmp_path / "2026-11-07-issue120-lifecycle.json"
    payload = (json.dumps(value) + "\n").encode()
    path.write_bytes(payload)
    write_trace(path.with_name(path.stem + "-power.csv"))

    def forbidden(*_a, **_k):
        pytest.fail("non-record companion reached measurement/trace analysis")

    monkeypatch.setattr(audit, "resolve_trace", forbidden)
    json_paths, _, _, analyses, records = audit._report_input_data(tmp_path)
    assert path in json_paths
    assert len(analyses) == 1
    result = analyses[0]
    assert result.status == f"non-record JSON companion ({kind})"
    assert records[path]["audit_json_root_type"] == kind
    assert result.resolution.path is None
    assert result.trace is None
    assert result.interval is None
    assert result.classification is None
    assert result.minimum_mhz is None
    assert result.maximum_mhz is None
    assert result.weighted_mhz is None
    assert result.weighted_is_exact is False
    assert result.timing_fields is False
    assert result.conversion_fields is False
    assert "excluded" in result.evidence
    assert path.read_bytes() == payload


def test_report_inventories_non_records_without_rendering_their_payload(tmp_path):
    path = tmp_path / "companion.json"
    path.write_text('[{"private_payload_marker": "ignore-me"}]\n')
    before = path.read_bytes()
    report = audit.render_report(tmp_path)
    assert "companion.json" in report
    assert "non-record JSON companion (array)" in report
    assert "association" in report
    assert "ignore-me" not in report
    assert "JSON files (records and companions): 1" in report
    assert path.read_bytes() == before


def test_mapping_records_keep_their_original_trace_analysis(tmp_path):
    path = tmp_path / "mapping.json"
    csv = tmp_path / "mapping-power.csv"
    digest = write_trace(csv)
    record = {"status": "complete", "power_trace": csv.name, "power_trace_sha256": digest}
    path.write_text(json.dumps(record) + "\n")
    before = path.read_bytes()
    _, _, _, analyses, records = audit._report_input_data(tmp_path)
    result = analyses[0]
    assert records[path] == record
    assert result.status == "complete"
    assert result.resolution.path == csv
    assert result.classification.unique_mhz == (800.0, 1350.0)
    assert path.read_bytes() == before


def test_malformed_json_keeps_existing_unreadable_classification(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_bytes(b"{broken")
    _, _, _, analyses, _ = audit._report_input_data(tmp_path)
    assert analyses[0].status == "unreadable: JSONDecodeError"
    assert path.read_bytes() == b"{broken"


def test_committed_lifecycle_arrays_are_preserved_and_classified():
    root = Path("docs/measurements")
    paths = [
        root / "2026-11-07-p150a-issue120-p1-launch-failure-lifecycle.json",
        root / "2026-11-07-p150a-issue120-p1-runtime-root-v1-failure-lifecycle.json",
    ]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    _, _, _, analyses, _ = audit._report_input_data(root)
    by_path = {item.path: item for item in analyses}
    for path in paths:
        assert by_path[path].status == "non-record JSON companion (array)"
        assert by_path[path].trace is None
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    assert after == before
