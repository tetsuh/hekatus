"""P2 must require the exact admitted P1 origin, not merely a matching UUID."""
from unittest.mock import patch

import pytest

from enodia.tt.bench.issue120_production import production_admission as production


@pytest.mark.parametrize("change", ["session_manifest_sha256", "program_identity", "environment", "paired"])
def test_p1_release_rejects_changed_origin_or_clock_anomalies(change):
    manifest = {"session_id": "a" * 32, "program_identity": {"native": "pinned"}, "harness_commit": "a" * 40}
    row = {"session_id": manifest["session_id"], "session_manifest_sha256": "a" * 64,
           "program_identity": manifest["program_identity"], "environment": {"commit": manifest["harness_commit"]},
           "metrics": {"paired_outlier_count": 0}}
    if change == "paired":
        row["metrics"]["paired_outlier_count"] = 1
    elif change == "environment":
        row[change] = {"commit": "different"}
    else:
        row[change] = "different"
    with (patch.object(production.sessions, "read_session", return_value=(manifest, "a" * 64)),
          pytest.raises(ValueError)):
        production.control_release(row, manifest["session_id"])
