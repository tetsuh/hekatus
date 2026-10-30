"""Board-free CLI coverage for the ADR settled-status guard."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools/adr_status_guard.py"
ACCEPTED = "- Status: **Accepted** — 2026-08-12\n"


def run_guard(root=None, *, cwd=ROOT):
    command = [sys.executable, str(SCRIPT)]
    if root is not None:
        command += ["--root", str(root)]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=10, check=False)


def write_adr(root, name, status):
    path = root / name
    path.write_text(f"# ADR: fixture\n\n{status}\n## Context\n\nFixture.\n", encoding="utf-8")
    return path


def assert_failure(result, path, reason):
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert str(path) in output
    assert reason in output
    assert "adr status guard: clean" not in output


@pytest.mark.parametrize("state", ["Accepted", "Rejected", "Deprecated", "Superseded by ADR-0003"])
def test_settled_state_passes(tmp_path, state):
    write_adr(tmp_path, "0001-fixture.md", f"- Status: **{state}** — 2026-08-12\n")
    result = run_guard(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "adr status guard: clean\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("- Status: **Proposed** — 2026-08-12\n", "Proposed"),
        ("", "missing status line"),
        ("- Status: **Unknown** — 2026-08-12\n", "unknown state"),
        (ACCEPTED + ACCEPTED, "multiple status lines"),
        (ACCEPTED + "- Status: **Proposed** — 2026-08-12\n", "multiple status lines"),
        ("- Status: Accepted — 2026-08-12\n", "malformed status line"),
        ("- Status: **Accepted** - 2026-08-12\n", "malformed status line"),
        ("- Status: **Accepted** — yesterday\n", "malformed status line"),
        ("- Status: **Accepted** — 2026-02-30\n", "invalid status date"),
        ("- Status: **Accepted** — 2026-8-12\n", "malformed status line"),
        ("- Status: **Accepted** — 2026-08-12 extra\n", "malformed status line"),
        ("- Status: **accepted** — 2026-08-12\n", "unknown state"),
        ("- Status: **Superseded** — 2026-08-12\n", "unknown state"),
        ("- Status: **Superseded by ADR-003** — 2026-08-12\n", "unknown state"),
        ("- Status: **Superseded by ADR-00003** — 2026-08-12\n", "unknown state"),
        ("- Status: **Superseded by ADR-０００３** — 2026-08-12\n", "unknown state"),
        ("  - Status: **Accepted** — 2026-08-12\n", "malformed status line"),
    ],
)
def test_unsettled_or_invalid_status_fails(tmp_path, status, reason):
    path = write_adr(tmp_path, "0001-fixture.md", status)
    assert_failure(run_guard(tmp_path), path, reason)


def test_status_must_be_in_header(tmp_path):
    path = tmp_path / "0001-fixture.md"
    path.write_text("# ADR\n\n## Context\n\n" + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


def test_duplicate_in_body_is_not_hidden(tmp_path):
    path = write_adr(tmp_path, "0001-fixture.md", ACCEPTED)
    with path.open("a") as stream:
        stream.write("\n" + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "multiple status lines")


def test_all_markdown_files_are_scanned_and_all_failures_reported(tmp_path):
    good = write_adr(tmp_path, "0001-good.md", ACCEPTED)
    proposed = write_adr(tmp_path, "9999-proposed-fixture.md", ACCEPTED.replace("Accepted", "Proposed"))
    missing = write_adr(tmp_path, "unnumbered.md", "")
    (tmp_path / "ignored.txt").write_text("not an ADR")
    result = run_guard(tmp_path)
    assert_failure(result, proposed, "Proposed")
    assert_failure(result, missing, "missing status line")
    assert str(good) not in result.stdout
    assert len(result.stdout.splitlines()) == 2


def test_only_current_tree_matters_without_git_or_history(tmp_path):
    path = write_adr(tmp_path, "0001-fixture.md", ACCEPTED.replace("Accepted", "Proposed"))
    assert_failure(run_guard(tmp_path), path, "Proposed")
    path.write_text(path.read_text().replace("Proposed", "Rejected"))
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
def test_unscannable_root_fails(tmp_path, kind):
    root = tmp_path / "adr"
    if kind == "empty":
        root.mkdir()
    elif kind == "file":
        root.write_text("not a directory")
    assert_failure(run_guard(root), root, "unable to scan ADR directory")


def test_unreadable_record_fails(tmp_path):
    path = tmp_path / "0001-invalid-utf8.md"
    path.write_bytes(b"\xff")
    assert_failure(run_guard(tmp_path), path, "unable to read ADR")


def test_default_root_is_relative_to_checkout(tmp_path):
    root = tmp_path / "docs/adr"
    root.mkdir(parents=True)
    write_adr(root, "0001-fixture.md", ACCEPTED)
    result = run_guard(cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "adr status guard: clean\n"


def test_repository_adrs_pass():
    result = run_guard()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "adr status guard: clean\n"
