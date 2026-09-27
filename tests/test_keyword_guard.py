"""Behavioral coverage for the shell run by the keyword-guard workflow."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

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


def _assert_no_configured_terms(output: str, words: list[str]) -> None:
    for word in words:
        assert word.casefold() not in output.casefold()


def _run_scan(
    tmp_path: Path,
    *,
    words: str | None,
    tracked: dict[str, str],
    timeout: float = 10,
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
    environment["LC_ALL"] = "C.UTF-8"
    if words is None:
        environment.pop("WORDS", None)
    else:
        environment["WORDS"] = words
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-e"],
        cwd=repo,
        env=environment,
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def test_missing_word_list_fails_with_actionable_message(tmp_path):
    completed = _run_scan(tmp_path, words=None, tracked={"clean.txt": "safe\n"})

    assert completed.returncode != 0
    assert "secret is not set" in completed.stdout
    assert "repository secret" in completed.stdout
    assert "configure it before merging" in completed.stdout


def test_word_list_with_only_separators_and_whitespace_fails(tmp_path):
    completed = _run_scan(tmp_path, words=" , \t,\n ", tracked={"clean.txt": "safe\n"})

    assert completed.returncode != 0
    assert "word list is empty" in completed.stdout
    assert "repository secret" in completed.stdout
    assert "set the repository secret" in completed.stdout


@pytest.mark.parametrize("whitespace", ["\u00a0", "\u2003", "\u3000"])
def test_word_list_with_only_unicode_whitespace_fails(tmp_path, whitespace):
    completed = _run_scan(
        tmp_path,
        words=f",{whitespace},",
        tracked={"clean.txt": "safe\n"},
    )

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


def test_clean_tree_with_absent_word_terminates_under_ci_shell(tmp_path):
    completed = _run_scan(
        tmp_path,
        words="not-present",
        tracked={"clean.txt": "safe\n"},
        timeout=2,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines() == ["keyword guard: clean"]


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
    _assert_no_configured_terms(output, [word])


def test_path_hit_is_masked_literally_when_word_contains_regex_metacharacters(tmp_path):
    word = "Ac.Me"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"notes/ac.me-record.txt": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word in path notes/ -record.txt (1 occurrence(s))" in output
    _assert_no_configured_terms(output, [word])


def test_skeleton_substrings_and_replacement_markers_never_leak(tmp_path):
    words = ["err", "rror", "fi", "ile", ":", "::", "*", "**", "***"]
    for directory, word in enumerate(words):
        cases = [
            ("clean", {"safe.txt": "safe\n"}, 0),
            ("contents", {"safe.txt": f"before {word} after\n"}, 1),
            ("path", {f"{word}.txt": "safe\n"}, 1),
        ]
        for mode, tracked, expected_returncode in cases:
            run_path = tmp_path / f"skeleton-{directory}-{mode}"
            run_path.mkdir()
            completed = _run_scan(run_path, words=word, tracked=tracked)

            output = completed.stdout + completed.stderr
            assert completed.returncode == expected_returncode
            _assert_no_configured_terms(output, [word])
            if mode == "clean":
                assert "keyword guard" in output
            else:
                assert "prohibited word" in output

            # These terms occur inside the command syntax itself. Their hit
            # reports must be one plain masked line rather than a command.
            if word in {"err", "rror", "fi", "ile", ":", "::"}:
                assert "::" not in output
                assert len(output.splitlines()) == 1


def test_configured_terms_cannot_cross_rendered_annotation_boundaries(tmp_path):
    cases = [
        ("::keyword", {"clean.txt": "safe\n"}, 0),
        ("=safe", {"safe.txt": "=safe\n"}, 1),
        ("txt::prohibited", {"safe.txt": "txt::prohibited\n"}, 1),
        ("::prohibited", {"::prohibited.txt": "safe\n"}, 1),
    ]

    for directory, (word, tracked, expected_returncode) in enumerate(cases):
        run_path = tmp_path / f"boundary-{directory}"
        run_path.mkdir()
        completed = _run_scan(run_path, words=word, tracked=tracked)

        output = completed.stdout + completed.stderr
        assert completed.returncode == expected_returncode
        assert len(output.splitlines()) == 1
        assert not output.startswith("::")
        assert "::error" not in output
        _assert_no_configured_terms(output, [word])


def test_unicode_case_equivalent_path_hits_are_masked_with_grep_semantics(tmp_path):
    cases = [
        ("σ", "docs/ο-ς.txt", "docs/ο- .txt"),
        ("s", "dir/ſ-afe.txt", "dir/ -afe.txt"),
    ]

    for directory, (word, path, masked_path) in enumerate(cases):
        run_path = tmp_path / f"unicode-{directory}"
        run_path.mkdir()
        completed = _run_scan(run_path, words=word, tracked={path: "safe\n"})

        output = completed.stdout + completed.stderr
        assert completed.returncode != 0
        assert f"prohibited word in path {masked_path}" in output
        _assert_no_configured_terms(output, [word])


def test_configured_terms_are_masked_in_fixed_reporter_messages(tmp_path):
    cases = [
        ("clean", {"safe.txt": "safe\n"}),
        ("contents", {"safe.txt": "contents\n"}),
    ]

    for directory, (word, tracked) in enumerate(cases):
        run_path = tmp_path / f"reporter-{directory}"
        run_path.mkdir()
        completed = _run_scan(run_path, words=word, tracked=tracked)

        output = completed.stdout + completed.stderr
        assert completed.returncode == (0 if word == "clean" else 1)
        _assert_no_configured_terms(output, [word])
        if word == "clean":
            assert output.splitlines() == ["keyword guard:  "]
        else:
            assert "prohibited word in" in output
            assert "(1 occurrence(s))" in output


def test_repeated_path_matches_terminate_and_mask_every_occurrence(tmp_path):
    word = "x"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"x-x.log": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word in path  - .log (2 occurrence(s))" in output
    _assert_no_configured_terms(output, [word])


def test_repeated_reporter_matches_terminate_and_mask_every_occurrence(tmp_path):
    word = "s"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"clean.txt": "s\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word" in output
    _assert_no_configured_terms(output, [word])


def test_broken_working_tree_symlink_scan_failure_is_not_clean(tmp_path):
    word = "needle"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "broken-link").symlink_to("missing-target")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)

    _, script = _workflow_step("Scan tracked files for prohibited words")
    environment = dict(os.environ)
    environment["LC_ALL"] = "C.UTF-8"
    environment["WORDS"] = word
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e"],
        cwd=repo,
        env=environment,
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "unable to scan tracked contents" in output
    assert "keyword guard: clean" not in output
    _assert_no_configured_terms(output, [word])


def test_git_ls_files_failure_is_not_clean(tmp_path):
    word = "needle"
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "clean.txt").write_text("safe\n")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)

    real_git = shutil.which("git")
    assert real_git is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "ls-files" ]; then\n'
        '  printf "clean.txt\\0"\n'
        "  exit 23\n"
        "fi\n"
        f'exec "{real_git}" "$@"\n'
    )
    fake_git.chmod(0o755)

    _, script = _workflow_step("Scan tracked files for prohibited words")
    environment = dict(os.environ)
    environment["LC_ALL"] = "C.UTF-8"
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["WORDS"] = word
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e"],
        cwd=repo,
        env=environment,
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "keyword guard: unable to enumerate tracked files" in output
    assert "keyword guard: clean" not in output
    _assert_no_configured_terms(output, [word])


def test_reserved_error_word_falls_back_to_fully_masked_plain_text(tmp_path):
    word = "error"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"error.txt": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "::" not in output
    assert "prohibited word in path" in output
    assert " .txt" in output
    _assert_no_configured_terms(output, [word])


def test_reserved_file_word_falls_back_to_fully_masked_plain_text(tmp_path):
    word = "file"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"file.txt": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "::" not in output
    assert "prohibited word in path" in output
    assert " .txt" in output
    _assert_no_configured_terms(output, [word])


def test_fixed_reporter_occurrence_word_is_masked_without_leaking(tmp_path):
    word = "occurrence"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={"safe.txt": f"{word}\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "prohibited word in contents" in output
    assert "(1  (s))" in output
    _assert_no_configured_terms(output, [word])


def test_fork_pull_request_rejection_stays_fail_closed_with_existing_message(tmp_path):
    header, script = _workflow_step("Reject unscannable fork pull requests")
    assert "github.event.pull_request.head.repo.full_name != github.repository" in header

    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e"],
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
