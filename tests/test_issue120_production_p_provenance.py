"""Native-bound production shape cannot downgrade to the historical reader."""
import pytest

from enodia.tt.bench.issue120_production import host_trace, recording


@pytest.mark.parametrize("metadata", [{}, {"phase": "calibration", "plan_identity": "production-v1-calibration-small"}])
def test_traced_production_cannot_delete_its_provenance(tmp_path, metadata):
    row = {"native": {"schema": host_trace.NATIVE_SCHEMA, "frame_count": 500000}, "program_identity": {}, **metadata}
    with pytest.raises(ValueError, match="production"):
        recording.recheck_calibration_provenance(tmp_path, row)
