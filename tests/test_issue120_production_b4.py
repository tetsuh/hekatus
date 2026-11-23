"""Renewed B4 identities do not change the measured program or budgets."""
from __future__ import annotations

from dataclasses import replace

import pytest

from enodia.tt.bench.issue120_production import plan, records, sessions


@pytest.mark.parametrize("condition,old,new", [
    ("control", "production-v1-control-watch", "production-v2-control-watch-b4"),
    ("loaded", "production-v1-load-watch", "production-v2-load-watch-b4"),
])
def test_b4_launch_identity_is_new_but_configuration_is_unchanged(condition, old, new):
    spec = plan.run_spec(new)
    assert spec in sessions.admitted_specs("B")
    assert old not in [row.identity for row in plan.RUNS]
    assert records.SCHEMA == "issue120-production-v2"
    assert spec.condition == condition
    assert spec.phase == "startup"
    assert (spec.frame_count, spec.repeat_count, spec.cap_seconds, spec.watcher) == (4096, 262144, 60, True)
    legacy = replace(spec, identity=old)
    assert plan.program_configuration(spec) == plan.program_configuration(legacy)
    assert plan.configuration_digest(plan.program_configuration(spec)) == plan.configuration_digest(
        plan.program_configuration(legacy))
