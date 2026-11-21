"""Real P admission/session/launcher/publication using only isolated CPU children."""
import pytest

from tools import issue120_production_dry_run


@pytest.mark.slow
def test_p_pair_cpu_e2e_and_fail_closed_boundaries():
    result = issue120_production_dry_run.run()
    assert result["fixture_only"] is True
    assert result["P_executed"] is False
    assert result["native_invoked"] is False
    assert result["positive_launches"] == 2
    assert result["P1_exit"] == result["P2_exit"] == 0
    assert result["comparison_recomputed"] is True
    for key in ("boot", "head", "missing_receipt", "modified_receipt", "missing_P1", "failed_P1", "missing_approval", "dirty_source", "environment_summary", "historical_P1", "missing_adoption", "mismatched_adoption", "used_roles"):
        assert result["rejected"][key] is True
