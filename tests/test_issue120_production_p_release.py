"""P2 release refusals start from an admitted, healthy P1 origin."""
from unittest.mock import patch

import pytest
from test_issue120_production_p_review import valid_origin

from enodia.tt.bench.issue120_production import production_admission as production


@pytest.mark.parametrize("change", ["session_manifest_sha256", "program_identity", "environment", "paired"])
def test_p1_release_rejects_changed_origin_or_clock_anomalies(change):
    manifest, row = valid_origin()
    with patch.object(production.sessions, "read_session", return_value=(manifest, "a" * 64)):
        production.control_release(row, manifest["session_id"])
        if change == "paired":
            row["metrics"]["paired_outlier_count"] = 1
        elif change == "environment":
            row["environment"]["commit"] = "different"
        elif change == "program_identity":
            row["program_identity"]["production_admission_contract"] = "different"
        else:
            row[change] = "different"
        with pytest.raises(ValueError, match="^production control role/origin or paired-clock anomalies require owner review$"):
            production.control_release(row, manifest["session_id"])
