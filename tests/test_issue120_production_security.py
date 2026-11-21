"""CPU-only rejection at generator and subprocess argument boundaries."""
from __future__ import annotations

import pytest

from enodia.tt.bench.issue120_production import kernel_adapter, plan, runner


def test_generator_cli_cannot_select_an_arbitrary_output(tmp_path):
    with pytest.raises(SystemExit):
        kernel_adapter.main([str(tmp_path / "unowned")])


def test_generator_rejects_unowned_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_adapter, "OWNED_OUTPUTS", (tmp_path / "owned",), raising=False)
    with pytest.raises(ValueError):
        kernel_adapter.generate(tmp_path / "outside")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("name", ["production_producer.cpp", "sources.json"])
def test_generator_rejects_redirected_targets(tmp_path, monkeypatch, name):
    owned = tmp_path / "owned"
    owned.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("unchanged")
    (owned / name).symlink_to(victim)
    monkeypatch.setattr(kernel_adapter, "OWNED_OUTPUTS", (owned,), raising=False)
    with pytest.raises(ValueError):
        kernel_adapter.generate(owned)
    assert victim.read_text() == "unchanged"


@pytest.mark.parametrize("identity", ["unit-session", "0"*31, "F"*32,
                                       "-" + "0"*31, "../escape", "0"*32 + " --run"])
def test_wrapper_rejects_noncanonical_session_arguments(identity):
    with pytest.raises(ValueError):
        runner.wrapper_command(plan.RUNS[0], identity)
