"""Behavioral coverage for the shell run by the keyword-guard workflow."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/keyword-guard.yml"


def _workflow_step(step_name: str) -> tuple[str, str]:
    """Return a step's YAML header and de-indented ``run`` body."""
    lines = WORKFLOW.read_text().splitlines()
    name_line = f"      - name: {step_name}"
    name_index = lines.index(name_line)
    run_index = next(
        index
        for index in range(name_index + 1, len(lines))
        if lines[index] == "        run: |"
    )

    body = []
    for line in lines[run_index + 1 :]:
        if line and not line.startswith("          "):
            break
        body.append(line[10:] if line else "")
    return "\n".join(lines[name_index:run_index]), "\n".join(body) + "\n"


def _run_scan(
    tmp_path: Path, *, words: str | None, tracked: dict[str, str]
) -> subprocess.CompletedProcess:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for relative_path, contents in tracked.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)

    _, script = _workflow_step("Scan tracked files for prohibited words")
    environment = dict(os.environ)
    if words is None:
        environment.pop("WORDS", None)
    else:
        environment["WORDS"] = words
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail"],
        cwd=repo,
        env=environment,
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_missing_word_list_fails_with_actionable_message(tmp_path):
    completed = _run_scan(tmp_path, words=None, tracked={"clean.txt": "safe\n"})

    assert completed.returncode != 0
    assert "secret is not set" in completed.stdout
    assert "repository secret" in completed.stdout


def test_word_list_with_only_separators_and_whitespace_fails(tmp_path):
    completed = _run_scan(tmp_path, words=" , \t,\n ", tracked={"clean.txt": "safe\n"})

    assert completed.returncode != 0
    assert "word list is empty" in completed.stdout
    assert "repository secret" in completed.stdout


def test_valid_word_list_scans_tracked_files_and_passes_a_clean_tree(tmp_path):
    completed = _run_scan(
        tmp_path,
        words="Ac.me",
        tracked={"clean.txt": "safe\n"},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "keyword guard: clean" in completed.stdout


def test_valid_word_list_fails_without_disclosing_content_hit(tmp_path):
    word = "Ac.me"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"content.txt": f"before {word} after\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word in contents (1 occurrence(s))" in output
    assert word.casefold() not in output.casefold()


def test_path_hit_is_masked_literally_when_word_contains_regex_metacharacters(tmp_path):
    word = "Ac.Me"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"notes/ac.me-record.txt": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word in path notes/***-record.txt (1 occurrence(s))" in output
    assert word.casefold() not in output.casefold()


def test_fork_pull_request_rejection_stays_fail_closed_with_existing_message(tmp_path):
    header, script = _workflow_step("Reject unscannable fork pull requests")
    assert "github.event.pull_request.head.repo.full_name != github.repository" in header

    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail"],
        cwd=tmp_path,
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 1
    assert completed.stdout.splitlines() == [
        "::error::Fork pull requests cannot be scanned by the keyword guard.",
        "Re-run the branch from a branch in this repository before merging.",
    ]
