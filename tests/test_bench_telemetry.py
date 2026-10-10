"""Telemetry parsing, kept out of shell quoting where it cannot be tested.

The first run of the harness produced a power trace containing only its
header: the sampler had been embedded in the wrapper script, and its
quoting was wrong. The parsing lives here instead, where it is exercised.
"""

import json
from pathlib import Path

import pytest

from enodia.tt.bench import telemetry
from enodia.tt.bench.telemetry import (
    parse_environment,
    parse_telemetry,
    sampler_metadata,
    telemetry_csv_row,
)

SNAPSHOT = json.dumps(
    {
        "device_info": [
            {
                "board_info": {"board_type": "p100a", "bus_id": "0000:01:00.0"},
                "firmwares": {"fw_bundle_version": "19.4.1.0"},
                "limits": {"tdp_limit": "150"},
                "telemetry": {
                    "power": " 37.0",
                    "aiclk": " 800",
                    "asic_temperature": "56.8",
                },
            }
        ]
    }
)


def test_telemetry_is_parsed_and_stripped():
    reading = parse_telemetry(SNAPSHOT)

    assert reading == {"power_w": "37.0", "aiclk_mhz": "800", "asic_temp_c": "56.8"}


def test_a_row_carries_a_timestamp_and_the_reading_in_column_order():
    row = telemetry_csv_row(SNAPSHOT, timestamp="2026-08-10T16:08:28+00:00")

    assert row == "2026-08-10T16:08:28+00:00,37.0,800,56.8"


def test_unparseable_telemetry_yields_no_row_rather_than_a_broken_one():
    assert parse_telemetry("not json at all") is None
    assert parse_telemetry(json.dumps({"device_info": []})) is None
    assert telemetry_csv_row("not json at all", timestamp="t") is None


def test_environment_keeps_the_identity_of_the_board_and_its_firmware():
    env = parse_environment(SNAPSHOT)

    assert env["board"]["board_type"] == "p100a"
    assert env["firmware"]["fw_bundle_version"] == "19.4.1.0"
    assert env["limits"]["tdp_limit"] == "150"
    assert env["aiclk_mhz_observed"] == [800]


def test_sampler_metadata_distinguishes_default_explicit_and_off_modes():
    assert sampler_metadata("default") == {
        "mode": "default",
        "interval_seconds": 2.0,
        "power_trace": "required",
        "timing_evidence": "available",
    }
    assert sampler_metadata("explicit", 5.0)["interval_seconds"] == 5.0
    assert sampler_metadata("off")["power_trace"] == "absent_by_design"
    assert sampler_metadata("off")["timing_evidence"] == "diagnostic_only"
    with pytest.raises(ValueError, match="default sampler"):
        sampler_metadata("default", 5.0)
    with pytest.raises(ValueError, match="off"):
        sampler_metadata("off", 5.0)


def test_capture_environment_records_sampler_mode(monkeypatch):
    monkeypatch.setattr(telemetry, "_run", lambda command: SNAPSHOT)
    environment = telemetry.capture_environment(
        "image@sha256:" + "a" * 64,
        True,
        sampler_mode="explicit",
        sampler_interval=5.0,
    )

    assert environment["telemetry_sampler"] == {
        "mode": "explicit",
        "interval_seconds": 5.0,
        "power_trace": "required",
        "timing_evidence": "available",
    }


def _two_board_snapshot() -> str:
    return json.dumps(
        {
            "device_info": [
                {
                    "board_info": {
                        "board_type": "p150a",
                        "board_id": "board-zero",
                        "bus_id": "0000:01:00.0",
                    },
                    "telemetry": {
                        "power": "10",
                        "aiclk": "800",
                        "asic_temperature": "50",
                    },
                },
                {
                    "board_info": {
                        "board_type": "p150a",
                        "board_id": "board-one",
                        "bus_id": "0000:02:00.0",
                    },
                    "telemetry": {
                        "power": "20",
                        "aiclk": "900",
                        "asic_temperature": "60",
                    },
                },
            ]
        }
    )


def test_explicit_numeric_node_uses_a_verified_pci_mapping(monkeypatch):
    monkeypatch.setenv("HEKATUS_TT_DEVICE_NODE", "/dev/tenstorrent/1")
    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda node: {"0000:02:00.0"},
    )

    assert telemetry.parse_telemetry(_two_board_snapshot()) == {
        "power_w": "20",
        "aiclk_mhz": "900",
        "asic_temp_c": "60",
    }
    environment = telemetry.parse_environment(_two_board_snapshot())
    assert environment["device_node"] == "/dev/tenstorrent/1"
    assert environment["board"]["board_id"] == "board-one"


def test_default_device_node_keeps_first_board_behavior(monkeypatch):
    monkeypatch.delenv("HEKATUS_TT_DEVICE_NODE", raising=False)

    assert telemetry.parse_environment(_two_board_snapshot())["board"]["board_id"] == "board-zero"


def test_unverifiable_or_ambiguous_explicit_node_is_rejected(monkeypatch):
    monkeypatch.setenv("HEKATUS_TT_DEVICE_NODE", "/dev/tenstorrent/1")
    monkeypatch.setattr(telemetry, "_resolve_device_node_to_pci", lambda node: set())
    assert telemetry.parse_telemetry(_two_board_snapshot()) is None

    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda node: {"0000:01:00.0", "0000:02:00.0"},
    )
    assert telemetry.parse_telemetry(_two_board_snapshot()) is None

    conflicting = json.loads(_two_board_snapshot())
    conflicting["device_info"][1]["board_info"]["pci_bdf"] = "0000:03:00.0"
    monkeypatch.setattr(
        telemetry,
        "_resolve_device_node_to_pci",
        lambda node: {"0000:02:00.0"},
    )
    assert telemetry.parse_telemetry(json.dumps(conflicting)) is None


def test_numeric_node_does_not_fall_back_to_snapshot_position(monkeypatch):
    monkeypatch.setenv("HEKATUS_TT_DEVICE_NODE", "/dev/tenstorrent/1")
    monkeypatch.setattr(telemetry, "_resolve_device_node_to_pci", lambda node: set())

    snapshot = json.loads(_two_board_snapshot())
    snapshot["device_info"][0]["device_id"] = 1
    assert telemetry.parse_environment(json.dumps(snapshot))["board_snapshot_error"]


def test_resolver_accepts_only_a_real_by_id_symlink(tmp_path):
    device_dir = tmp_path / "tenstorrent"
    by_id = device_dir / "by-id"
    by_id.mkdir(parents=True)
    node = device_dir / "1"
    node.write_text("")
    link = by_id / "pci-0000:02:00.0"
    link.symlink_to("../1")
    identity = lambda path: {"0000:02:00.0"}

    assert telemetry._resolve_device_node_to_pci(str(node), identity_resolver=identity) == {
        "0000:02:00.0"
    }
    assert telemetry._resolve_device_node_to_pci(str(link), identity_resolver=identity) == {
        "0000:02:00.0"
    }


def test_resolver_accepts_blackhole_by_id_link_from_target_identity(tmp_path):
    device_dir = tmp_path / "tenstorrent"
    by_id = device_dir / "by-id"
    by_id.mkdir(parents=True)
    node = device_dir / "1"
    node.write_text("")
    link = by_id / "blackhole-80A3FF7BA86938B3"
    link.symlink_to("../1")
    seen_targets = []

    def resolve_identity(path):
        seen_targets.append(path)
        return {"0000:09:00.0"}

    assert telemetry._resolve_device_node_to_pci(str(link), identity_resolver=resolve_identity) == {
        "0000:09:00.0"
    }
    assert seen_targets == [node.resolve()]


def test_blackhole_by_id_selects_the_board_with_matching_snapshot_bus(tmp_path, monkeypatch):
    device_dir = tmp_path / "tenstorrent"
    by_id = device_dir / "by-id"
    by_id.mkdir(parents=True)
    node = device_dir / "1"
    node.write_text("")
    link = by_id / "blackhole-80A3FF7BA86938B3"
    link.symlink_to("../1")
    monkeypatch.setenv("HEKATUS_TT_DEVICE_NODE", str(link))
    monkeypatch.setattr(telemetry, "_device_pci_bus_ids", lambda path: {"0000:09:00.0"})

    snapshot = json.loads(_two_board_snapshot())
    snapshot["device_info"][0]["board_info"]["bus_id"] = "0000:06:00.0"
    snapshot["device_info"][1]["board_info"]["bus_id"] = "0000:09:00.0"

    assert telemetry.parse_telemetry(json.dumps(snapshot)) == {
        "power_w": "20",
        "aiclk_mhz": "900",
        "asic_temp_c": "60",
    }


def test_resolver_derives_numeric_node_identity_from_udev_devpath(tmp_path, monkeypatch):
    device_dir = tmp_path / "tenstorrent"
    device_dir.mkdir()
    node = device_dir / "1"
    node.write_text("")
    udev_path = "/devices/pci0000:00/0000:09:00.0/tenstorrent/1"
    monkeypatch.setattr(
        telemetry,
        "_udevadm_info",
        lambda path: (
            f"E: DEVPATH={udev_path}\n"
            f"E: ID_PATH=pci-0000:09:00.0\n"
            "E: PCI_SLOT_NAME=0000:09:00.0\n"
            f"E: DEVNAME={path}\n"
        ),
    )

    assert telemetry._resolve_device_node_to_pci(str(node)) == {"0000:09:00.0"}


def test_resolver_rejects_conflicting_udev_identities(tmp_path, monkeypatch):
    device_dir = tmp_path / "tenstorrent"
    device_dir.mkdir()
    node = device_dir / "1"
    node.write_text("")
    udev_path = "/devices/pci0000:00/0000:09:00.0/tenstorrent/1"
    monkeypatch.setattr(
        telemetry,
        "_udevadm_info",
        lambda path: (
            f"E: DEVPATH={udev_path}\n"
            "E: ID_PATH=pci-0000:06:00.0\n"
            "E: PCI_SLOT_NAME=0000:09:00.0\n"
            f"E: DEVNAME={path}\n"
        ),
    )

    assert telemetry._resolve_device_node_to_pci(str(node)) == set()


def test_resolver_rejects_conflicting_or_ambiguous_target_identity(tmp_path):
    device_dir = tmp_path / "tenstorrent"
    by_id = device_dir / "by-id"
    by_id.mkdir(parents=True)
    node = device_dir / "1"
    node.write_text("")
    link = by_id / "blackhole-80A3FF7BA86938B3"
    link.symlink_to("../1")

    assert (
        telemetry._resolve_device_node_to_pci(
            str(link), identity_resolver=lambda path: {"0000:06:00.0", "0000:09:00.0"}
        )
        == set()
    )
    assert (
        telemetry._resolve_device_node_to_pci(
            str(link), identity_resolver=lambda path: {"not-a-bdf"}
        )
        == set()
    )


def test_resolver_rejects_a_pci_label_that_conflicts_with_target_identity(tmp_path):
    device_dir = tmp_path / "tenstorrent"
    by_id = device_dir / "by-id"
    by_id.mkdir(parents=True)
    node = device_dir / "1"
    node.write_text("")
    link = by_id / "pci-0000:06:00.0"
    link.symlink_to("../1")

    assert (
        telemetry._resolve_device_node_to_pci(
            str(link), identity_resolver=lambda path: {"0000:09:00.0"}
        )
        == set()
    )


def test_environment_survives_a_snapshot_it_cannot_read():
    env = parse_environment("")

    assert "board_snapshot_error" in env
    assert "board" not in env


def test_environment_capture_carries_the_wrapper_run_id(monkeypatch):
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "captured-run")
    monkeypatch.setattr(telemetry, "_run", lambda command: "")
    monkeypatch.setattr(
        telemetry,
        "harness_identity",
        lambda: {"harness_commit": "a" * 40, "harness_dirty": False},
    )

    environment = telemetry.capture_environment("image@sha256:" + "a" * 64, True)

    assert environment["run_id"] == "captured-run"


@pytest.mark.parametrize(
    "snapshot",
    [
        json.dumps({"device_info": {}}),
        json.dumps({"device_info": {"board_info": {}}}),
        json.dumps({"device_info": 5}),
        json.dumps({"device_info": [5]}),
        json.dumps({"device_info": ["not a device"]}),
        json.dumps([]),
    ],
)
def test_a_malformed_snapshot_yields_nothing_rather_than_escaping(snapshot):
    """An exception here would kill the sampler, and a dead sampler is silent
    — which is exactly how the first run produced a trace with only a header."""
    assert parse_telemetry(snapshot) is None
    assert telemetry_csv_row(snapshot, timestamp="t") is None
    assert "board_snapshot_error" in parse_environment(snapshot)


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_sampling_intervals_are_rejected(interval, tmp_path):
    """Zero hammers the tool; negative and non-finite values reach sleep and
    end the sampler. Every one of them yields a trace nobody can use."""
    argv = ["sample", "--out", str(tmp_path / "p.csv"), "--interval", interval]

    with pytest.raises(SystemExit) as excinfo:
        telemetry.main(argv)

    assert excinfo.value.code == 2


class _WriterThatFillsUp:
    """Accepts the header, then fails the way a full disk does."""

    def __init__(self) -> None:
        self.writes = 0

    def write(self, text: str) -> int:
        self.writes += 1
        if self.writes == 1:
            return len(text)
        if self.writes > 20:
            raise AssertionError("the sampler kept retrying after a write failure")
        raise OSError("No space left on device")


def test_a_failing_writer_ends_the_sampler_rather_than_retrying(monkeypatch):
    """Surviving a bad sample is the point; surviving the inability to record
    anything is not — that turns a broken run into a quietly short trace."""
    monkeypatch.setattr(telemetry, "_run", lambda command: SNAPSHOT)

    writer = _WriterThatFillsUp()

    with pytest.raises(OSError, match="No space left"):
        telemetry._sample_loop(writer, interval=0.0)


def _fake_git(status_output="", *, seen=None):
    def fake_git(repo, *args):
        if seen is not None:
            seen.append(args)
        return ("abc1234def", True) if args[0] == "rev-parse" else (status_output, True)

    return fake_git


def test_the_environment_names_the_harness_that_produced_it(monkeypatch):
    """A result is only reproducible if the code that computed it can be named:
    the FLOP accounting behind every efficiency figure lives in this tree."""
    monkeypatch.setattr(telemetry, "_git", _fake_git())

    identity = telemetry.harness_identity()

    assert identity["harness_commit"] == "abc1234def"
    assert identity["harness_dirty"] is False


def test_transferred_tree_can_record_verified_harness_identity_without_git(monkeypatch):
    monkeypatch.setenv("HEKATUS_TT_HARNESS_COMMIT", "416a7fc")
    monkeypatch.setenv("HEKATUS_TT_HARNESS_DIRTY", "false")

    def fail_git(*args, **kwargs):
        raise AssertionError("explicit transferred-tree identity should bypass git")

    monkeypatch.setattr(telemetry, "_git", fail_git)

    assert telemetry.harness_identity() == {
        "harness_commit": "416a7fc",
        "harness_dirty": False,
        "harness_identity_source": "HEKATUS_TT_HARNESS_COMMIT",
    }


def test_a_modified_tree_is_recorded_as_such(monkeypatch):
    monkeypatch.setattr(telemetry, "_git", _fake_git(" M enodia/tt/bench/run_matmul.py"))

    assert telemetry.harness_identity()["harness_dirty"] is True


def test_untracked_files_do_not_mark_the_harness_modified(monkeypatch):
    """The toolchain writes build output into the tree on every run, so a flag
    that counts untracked files is true always and says nothing about the code
    that computed the result."""
    seen = []
    monkeypatch.setattr(telemetry, "_git", _fake_git(seen=seen))

    identity = telemetry.harness_identity()

    status = next(args for args in seen if args[0] == "status")
    assert "--untracked-files=no" in status
    assert identity["harness_dirty"] is False


@pytest.mark.parametrize("failing", ["rev-parse", "status"])
def test_a_git_query_that_fails_leaves_the_harness_unknown_rather_than_clean(failing, monkeypatch):
    """A failed query and a clean tree both put nothing on standard output.
    Told apart by the exit status only — and if they are not told apart, a
    harness nobody can name is recorded as an identified, unmodified one."""

    def fake_git(repo, *args):
        if args[0] == failing:
            return "git failed: fatal: detected dubious ownership", False
        return ("abc1234def", True) if args[0] == "rev-parse" else ("", True)

    monkeypatch.setattr(telemetry, "_git", fake_git)

    identity = telemetry.harness_identity()

    assert identity["harness_commit"] is None
    assert identity["harness_dirty"] is None
    assert "dubious ownership" in identity["harness_identity_error"]


def test_a_nonzero_exit_is_a_failure_even_with_empty_output(monkeypatch):
    """The exit status is the whole signal, so it is read from the process
    rather than inferred from what the process printed."""

    class _Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository\n"

    monkeypatch.setattr(telemetry.subprocess, "run", lambda *a, **k: _Failed())

    output, ok = telemetry._git(Path("/nowhere"), "rev-parse", "HEAD")

    assert ok is False
    assert "not a git repository" in output


def test_a_missing_git_is_a_failure_rather_than_an_exception(monkeypatch):
    """The sampler and the environment capture must survive a missing tool;
    what they must not do is report an identity they did not obtain."""

    def explode(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(telemetry.subprocess, "run", explode)

    output, ok = telemetry._git(Path("/nowhere"), "rev-parse", "HEAD")

    assert ok is False
    assert "FileNotFoundError" in output
