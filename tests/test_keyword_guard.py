"""Behavioral coverage for the shell run by the keyword-guard workflow."""

import bz2
import gzip
import io
import os
import re
import shutil
import struct
import subprocess
import tarfile
import zlib
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


@pytest.mark.parametrize("control_bytes", ["::", "%0A", "%0D", ",", "=", "\r", "\n"])
def test_command_control_bytes_in_tracked_paths_stay_inert(tmp_path, control_bytes):
    word = "needle"
    path = f"docs/{word}{control_bytes}report.txt"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={path: "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert len(output.splitlines()) == 1
    assert "::" not in output
    assert output.count("prohibited word in path") == 1
    assert "(1 occurrence(s))" in output
    _assert_no_configured_terms(output, [word])


@pytest.mark.parametrize("control_bytes", ["::", "%0A", "%0D", ",", "=", "\r", "\n"])
def test_command_control_bytes_in_annotated_contents_diagnostics_stay_inert(
    tmp_path, control_bytes
):
    word = "needle"
    path = f"docs/report{control_bytes}.txt"
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={path: f"safe {word}\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert len(output.splitlines()) == 1
    assert "::" not in output
    assert ": prohibited word in contents (1 occurrence(s))" in output
    assert "prohibited word in contents of" not in output
    assert "prohibited word in path" not in output
    assert "(1 occurrence(s))" in output
    _assert_no_configured_terms(output, [word])


@pytest.mark.parametrize(
    ("word", "expected_reports"),
    [
        ("foo::bar", 1),
        ("foo%0Abar", 1),
        ("foo%0Dbar", 1),
        # Commas delimit the configured list, so this path produces one
        # finding for each parsed term while retaining the unsafe separator.
        ("foo,bar", 2),
        ("foo=bar", 1),
    ],
)
def test_configured_terms_spanning_command_control_paths_stay_inert(
    tmp_path, word, expected_reports
):
    completed = _run_scan(
        tmp_path,
        words=word,
        tracked={f"docs/{word}.txt": "safe\n"},
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert len(output.splitlines()) == expected_reports
    assert "::" not in output
    assert output.count("prohibited word in path") == expected_reports
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


# Archive cases use only meaningless dummy terms and run the actual workflow shell.
ARCHIVE_WORD = "Zqxv.Dummy"


def _tar_bytes(members, *, compressed=True, archive_format=tarfile.USTAR_FORMAT):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=archive_format) as archive:
        for name, contents, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
            else:
                if kind in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    member.linkname = "target.txt"
                archive.addfile(member)
    data = buffer.getvalue()
    return gzip.compress(data, mtime=0) if compressed else data


def _run_archive_scan(tmp_path, data, *, name="evidence.tar.gz", words=ARCHIVE_WORD, limits=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / name).write_bytes(data)
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    _, script = _workflow_step("Scan tracked files for prohibited words")
    for key, value in (limits or {}).items():
        # Keep boundary tests small while executing the exact production logic.
        script, replacements = re.subn(rf"(?m)^{key}=\d+$", f"{key}={value}", script)
        assert replacements == 1
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-e"],
        cwd=repo,
        env={**os.environ, "LC_ALL": "C.UTF-8", "WORDS": words},
        input=script,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    output = completed.stdout + completed.stderr
    _assert_no_configured_terms(output, words.split(","))
    return completed.returncode, output


@pytest.mark.parametrize(
    "name", ["evidence.tar.gz", "evidence.tgz", "evidence.data"]
)
@pytest.mark.parametrize("member_name", ["notes.txt", "frames.bin"])
def test_archive_text_member_hit_names_archive_and_member(tmp_path, name, member_name):
    data = _tar_bytes(
        [(member_name, f"before {ARCHIVE_WORD.swapcase()} after\n".encode(), tarfile.REGTYPE)],
        compressed=True,
    )
    status, output = _run_archive_scan(tmp_path, data, name=name)
    assert status == 1
    assert name in output
    assert member_name in output
    assert "prohibited word in archive member contents (1 occurrence(s))" in output


def test_archive_container_only_hit_is_not_scanned(tmp_path):
    buffer = io.BytesIO()
    # FNAME deterministically places the dummy term in gzip's container header.
    with gzip.GzipFile(filename=ARCHIVE_WORD, fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(_tar_bytes([("notes.txt", b"safe\n", tarfile.REGTYPE)], compressed=False))
    data = buffer.getvalue()
    assert ARCHIVE_WORD.encode() in data
    assert ARCHIVE_WORD.encode() not in gzip.decompress(data)
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 0
    assert "scanned 1 text member(s)" in output


@pytest.mark.parametrize("nul_at_end", [False, True])
def test_archive_nul_member_is_skipped_and_listed(tmp_path, nul_at_end):
    payload = f" {ARCHIVE_WORD} ".encode()
    payload = payload + b"\0" if nul_at_end else b"\0" + payload
    data = _tar_bytes([("frames.bin", payload, tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 0
    assert "evidence.tar.gz" in output
    assert "frames.bin" in output
    assert "skipped binary member" in output
    assert "skipped 1 binary member(s)" in output


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE,
             tarfile.FIFOTYPE, tarfile.DIRTYPE, tarfile.AREGTYPE, tarfile.GNUTYPE_SPARSE, b"?"]
)
def test_archive_links_and_special_members_fail_closed(tmp_path, kind):
    data = _tar_bytes([("unsafe", b"", kind)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "evidence.tar.gz" in output
    assert "unsupported member type" in output


@pytest.mark.parametrize(
    "path",
    ["../escape.txt", "a/../escape.txt", "/absolute.txt", "C:/escape.txt", "a\\..\\escape.txt"],
)
def test_archive_escaping_paths_fail_closed(tmp_path, path):
    data = _tar_bytes([(path, b"safe\n", tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "unsafe member path" in output
    assert "evidence.tar.gz" in output
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize(
    ("limit", "value", "members", "message"),
    [
        (
            "MAX_ARCHIVE_MEMBERS", 1,
            [("a", b"a", tarfile.REGTYPE), ("b", b"b", tarfile.REGTYPE)],
            "member count limit",
        ),
        ("MAX_ARCHIVE_MEMBER_BYTES", 3, [("large", b"safe", tarfile.REGTYPE)], "member size limit"),
        (
            "MAX_ARCHIVE_TOTAL_BYTES", 6,
            [("a", b"safe", tarfile.REGTYPE), ("b", b"safe", tarfile.REGTYPE)],
            "total size limit",
        ),
        (
            "MAX_ARCHIVE_STREAM_BYTES", 512, [("a", b"safe", tarfile.REGTYPE)],
            "unreadable or invalid archive",
        ),
    ],
)
def test_archive_expansion_limits_fail_closed(tmp_path, limit, value, members, message):
    status, output = _run_archive_scan(tmp_path, _tar_bytes(members), limits={limit: value})
    assert status == 1
    assert message in output
    assert "evidence.tar.gz" in output


@pytest.mark.parametrize(
    "data",
    [b"not a tar archive", b"\x1f\x8bgarbage", _tar_bytes([("a", b"safe", tarfile.REGTYPE)])[:-6]],
)
def test_corrupt_archive_fails_closed(tmp_path, data):
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "unreadable or invalid archive" in output
    assert "evidence.tar.gz" in output


def test_archive_binary_member_path_is_still_scanned(tmp_path):
    member = f"records/{ARCHIVE_WORD}.txt"
    data = _tar_bytes([(member, b"\0safe", tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "prohibited word in archive member path" in output
    assert "evidence.tar.gz" in output
    assert "records/ .txt" in output


@pytest.mark.parametrize("name", ["nested.tar.gz", "opaque.bin"])
def test_nested_archive_is_rejected_before_binary_skip(tmp_path, name):
    nested = _tar_bytes([("safe", b"safe\n", tarfile.REGTYPE)])
    data = _tar_bytes([(name, nested, tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "nested archive is not supported" in output
    assert name in output
    assert "skipped binary member" not in output


@pytest.mark.parametrize("suffix", ["zip", "tar.xz", "gz", "bz2", "7z", "rar", "zst", "whl", "jar"])
def test_unhandled_archive_extensions_fail_closed(tmp_path, suffix):
    status, output = _run_archive_scan(tmp_path, b"safe", name=f"evidence.{suffix}")
    assert status == 1
    assert "unsupported archive format" in output


@pytest.mark.parametrize(
    "magic", [b"PK\x03\x04", b"\xfd7zXZ\x00", bz2.compress(b"safe"), b"7z\xbc\xaf\x27\x1c"]
)
def test_unhandled_archive_magic_fails_closed_without_extension(tmp_path, magic):
    status, output = _run_archive_scan(tmp_path, magic + b"safe", name="evidence.data")
    assert status == 1
    assert "unsupported archive format" in output


@pytest.mark.parametrize("control", ["::", "%0A", "%0D", ",", "="])
@pytest.mark.parametrize("binary", [False, True])
def test_archive_member_diagnostics_keep_control_bytes_inert(tmp_path, control, binary):
    member = f"records/{ARCHIVE_WORD}{control}report.txt"
    payload = b"\0safe" if binary else f" {ARCHIVE_WORD} ".encode()
    status, output = _run_archive_scan(tmp_path, _tar_bytes([(member, payload, tarfile.REGTYPE)]))
    assert status == 1  # Path hits must fail even when the member is binary.
    for line in output.splitlines():
        if "records/" in line:
            assert "::" not in line
            assert "%" not in line
    assert "records/ " in output


@pytest.mark.parametrize("word", ["σ", "s", "err", ":", "::", "*", "contents", "member"])
def test_archive_reports_preserve_existing_mask_semantics(tmp_path, word):
    member = "ο-ς.txt" if word == "σ" else "ſ-safe.txt" if word == "s" else "safe.txt"
    data = _tar_bytes([(member, f" {word} \n".encode(), tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data, words=word)
    assert status == 1
    assert output


def test_archive_text_uses_literal_word_boundaries_and_all_terms(tmp_path):
    payload = f"prefix{ARCHIVE_WORD}suffix ZqxvXDummy {ARCHIVE_WORD.lower()} second-dummy\n"
    data = _tar_bytes([("notes.txt", payload.encode(), tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data, words=f"{ARCHIVE_WORD},second-dummy")
    assert status == 1
    assert output.count("prohibited word in archive member contents (1 occurrence(s))") == 2


def test_non_archive_binary_contents_still_fail(tmp_path):
    status, output = _run_archive_scan(
        tmp_path, f"\0 {ARCHIVE_WORD} \n".encode(), name="frames.bin"
    )
    assert status == 1
    assert "prohibited word in contents (1 occurrence(s))" in output


def test_archive_limits_allow_exact_boundary(tmp_path):
    data = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)])
    status, output = _run_archive_scan(
        tmp_path, data,
        limits={
            "MAX_ARCHIVE_MEMBERS": 1,
            "MAX_ARCHIVE_MEMBER_BYTES": 4,
            "MAX_ARCHIVE_TOTAL_BYTES": 4,
        },
    )
    assert status == 0
    assert "scanned 1 text member(s)" in output


@pytest.mark.parametrize(
    "corruption", ["later-header", "crc", "isize", "missing-end", "trailer", "concatenated"]
)
def test_archive_corruption_cannot_masquerade_as_end_of_archive(tmp_path, corruption):
    raw = _tar_bytes(
        [("first", b"safe", tarfile.REGTYPE), ("second", b"safe", tarfile.REGTYPE)],
        compressed=False,
    )
    if corruption == "later-header":
        raw = raw[:1024] + b"!" + raw[1025:]  # Invalid second-header checksum.
    elif corruption == "missing-end":
        raw = raw[:2048]
    elif corruption == "trailer":
        raw += b"hidden payload"
    data = gzip.compress(raw, mtime=0)
    if corruption in {"crc", "isize"}:
        offset = -8 if corruption == "crc" else -4
        data = data[:offset] + bytes([data[offset] ^ 1]) + data[offset + 1:]
    elif corruption == "concatenated":
        data += gzip.compress(raw, mtime=0)
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "unreadable or invalid archive" in output


@pytest.mark.parametrize("member", ["nested.zip", "nested.tar", "nested.tar.xz"])
def test_nested_archive_suffix_is_rejected_even_for_text_payload(tmp_path, member):
    status, output = _run_archive_scan(tmp_path, _tar_bytes([(member, b"safe", tarfile.REGTYPE)]))
    assert status == 1
    assert "nested archive is not supported" in output


def test_archive_outer_path_still_scanned_and_masked(tmp_path):
    name = f"{ARCHIVE_WORD}.tar.gz"
    data = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data, name=name)
    assert status == 1
    assert "prohibited word in path  .tar.gz" in output


def test_archive_default_header_limit_fails_closed(tmp_path):
    data = _tar_bytes([(f"member-{index}", b"", tarfile.REGTYPE) for index in range(257)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "member count limit" in output


@pytest.mark.parametrize("prefix", [b"BZhang", b"BZh9safe", b"MSCFnotes", b"070701,", b"070702,"])
@pytest.mark.parametrize("has_word", [False, True])
@pytest.mark.parametrize("member", [False, True])
def test_weak_archive_prefixes_keep_ordinary_text_scanning(tmp_path, prefix, has_word, member):
    payload = prefix + (f" {ARCHIVE_WORD} \n".encode() if has_word else b" safe\n")
    data = _tar_bytes([("notes.txt", payload, tarfile.REGTYPE)]) if member else payload
    status, output = _run_archive_scan(
        tmp_path, data, name="evidence.tar.gz" if member else "ordinary.txt"
    )
    assert status == int(has_word)
    assert "unsupported archive format" not in output
    assert "nested archive" not in output
    if has_word:
        expected = "archive member contents" if member else "contents"
        assert f"prohibited word in {expected} (1 occurrence(s))" in output


@pytest.mark.parametrize("prefix", [b"BZhang", b"MSCFnotes", b"070701,", b"070702,"])
def test_weak_archive_prefixes_in_nul_members_are_named_skips(tmp_path, prefix):
    data = _tar_bytes([("frames.bin", prefix + f"\0 {ARCHIVE_WORD} ".encode(), tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 0
    assert "frames.bin: skipped binary member" in output


@pytest.mark.parametrize(
    "signature",
    [bz2.compress(b"safe"), bz2.compress(b""), b"MSCF\0\0\0\0", b"07070100000001", b"070702abcdef12"],
)
@pytest.mark.parametrize("member", [False, True])
def test_strong_unsupported_signatures_fail_without_suffix(tmp_path, signature, member):
    data = _tar_bytes([("opaque.bin", signature, tarfile.REGTYPE)]) if member else signature
    status, output = _run_archive_scan(
        tmp_path, data, name="evidence.tar.gz" if member else "opaque.data"
    )
    assert status == 1
    assert ("nested archive is not supported" if member else "unsupported archive format") in output


@pytest.mark.parametrize("binary", [False, True])
def test_nonzero_physical_padding_fails_before_member_classification(tmp_path, binary):
    payload = b"safe\0" if binary else b"safe"
    raw = _tar_bytes([("notes.txt", payload, tarfile.REGTYPE)], compressed=False)
    padding_start = 512 + len(payload)
    marker = f" {ARCHIVE_WORD} ".encode()
    raw = raw[:padding_start] + marker + raw[padding_start + len(marker):]
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 1
    assert "nonzero padding" in output
    assert "skipped binary member" not in output


def test_incomplete_final_zero_record_fails_closed(tmp_path):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False) + b"\0"
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 1
    assert "invalid archive end records" in output


def _rewrite_tar_header(raw, field_offset, value):
    header = bytearray(raw[:512])
    header[field_offset:field_offset + len(value)] = value
    header[148:156] = b" " * 8
    header[148:156] = f"{sum(header):06o}\0 ".encode()
    # Ensure a checksum failure cannot satisfy a reserved-field regression.
    tarfile.TarInfo.frombuf(bytes(header), "utf-8", "surrogateescape")
    return bytes(header) + raw[512:]


@pytest.mark.parametrize("offset", [20, 157, 275, 305, 360, 500])
def test_unused_tar_header_bytes_cannot_hide_terms(tmp_path, offset):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    raw = _rewrite_tar_header(raw, offset, b" Qzx ")
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0), words="Qzx")
    assert status == 1
    assert "invalid archive header field" in output


def test_numeric_header_nul_tail_cannot_hide_terms(tmp_path):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    raw = _rewrite_tar_header(raw, 136, b"1\0 Qzx\0     ")
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0), words="Qzx")
    assert status == 1
    assert "invalid archive header field" in output


@pytest.mark.parametrize("name", ["evidence.tar", "evidence.data"])
def test_raw_tar_is_outside_the_gzip_profile(tmp_path, name):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    status, output = _run_archive_scan(tmp_path, raw, name=name)
    assert status == 1
    assert "gzip" in output


@pytest.mark.parametrize("layout", ["GNU", "V7"])
def test_non_posix_layout_is_refused(tmp_path, layout):
    raw = _tar_bytes(
        [("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False,
        archive_format=tarfile.GNU_FORMAT if layout == "GNU" else tarfile.USTAR_FORMAT,
    )
    if layout == "V7":
        raw = _rewrite_tar_header(raw, 257, b"\0" * 255)
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 1
    assert "POSIX ustar" in output


@pytest.mark.parametrize("offset,width", [(100, 8), (108, 8), (116, 8), (124, 12), (136, 12), (329, 8), (337, 8)])
def test_base256_numeric_fields_are_refused(tmp_path, offset, width):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    raw = _rewrite_tar_header(raw, offset, b"\x80" + b"\0" * (width - 2) + b"\x04")
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 1
    assert "invalid archive header field" in output


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK])
def test_extension_records_are_refused_without_interpretation(tmp_path, kind):
    # Repeated PAX paths must never create a per-record process surface.
    payload = b"18 path=notes.txt\n" * 32 if kind in {tarfile.XHDTYPE, tarfile.XGLTYPE} else b"notes.txt\0"
    header = tarfile.TarInfo("././@Extension")
    header.type = kind
    header.size = len(payload)
    member = tarfile.TarInfo("notes.txt")
    member.size = 4
    raw = header.tobuf(format=tarfile.USTAR_FORMAT) + payload + b"\0" * (-len(payload) % 512)
    raw += member.tobuf(format=tarfile.USTAR_FORMAT) + b"safe" + b"\0" * 508 + b"\0" * 1024
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        assert len(list(archive)) == 1  # Valid broad-format input, not a broken fixture.
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 1
    assert "unsupported member type" in output


def test_directory_with_trailing_slash_is_refused(tmp_path):
    data = _tar_bytes([("Qzx/", b"", tarfile.DIRTYPE)])
    status, output = _run_archive_scan(tmp_path, data, words="Qzx/")
    assert status == 1
    assert "unsupported member type" in output


@pytest.mark.parametrize("control", ["\x01", "\x1f", "\x7f", "\t", "\r", "\n"])
def test_raw_member_path_controls_are_refused(tmp_path, control):
    data = _tar_bytes([(f"notes/{control}safe", b"safe", tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "unsafe member path" in output


def test_raw_ustar_prefix_and_name_are_scanned_without_normalization(tmp_path):
    member = "outer/" + "p" * 110 + "/./Qzx.txt"
    data = _tar_bytes([(member, b"safe", tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data, words="./Qzx")
    assert status == 1
    assert "prohibited word in archive member path" in output


def _gzip_metadata(raw, field, *, checksum=False):
    base = gzip.compress(raw, mtime=0)
    term = ARCHIVE_WORD.encode()
    if field == "extra":
        value = b"QZ" + struct.pack("<H", len(term)) + term
        optional = struct.pack("<H", len(value)) + value
        flag = 4
    else:
        optional = term + b"\0"
        flag = 8 if field == "filename" else 16
    header = bytearray(base[:10])
    header[3] = flag | (2 if checksum else 0)
    header += optional
    checksum_offset = len(header)
    if checksum:
        header += struct.pack("<H", zlib.crc32(header) & 0xffff)
    return bytes(header) + base[10:], checksum_offset


@pytest.mark.parametrize("field", ["extra", "filename", "comment"])
@pytest.mark.parametrize("checksum", [False, True])
def test_gzip_optional_fields_are_intentional_container_metadata(tmp_path, field, checksum):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    data, _ = _gzip_metadata(raw, field, checksum=checksum)
    assert ARCHIVE_WORD.encode() in data
    assert ARCHIVE_WORD.encode() not in raw
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 0
    assert "scanned 1 text member(s)" in output


@pytest.mark.parametrize("field", ["extra", "filename", "comment"])
def test_gzip_header_checksum_cannot_be_ignored(tmp_path, field):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    data, offset = _gzip_metadata(raw, field, checksum=True)
    data = data[:offset] + bytes([data[offset] ^ 1]) + data[offset + 1:]
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "invalid archive gzip envelope" in output


@pytest.mark.parametrize("tail", [gzip.compress(b"", mtime=0), b"\0"])
def test_extra_gzip_stream_or_zero_tail_is_rejected(tmp_path, tail):
    data = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)]) + tail
    status, output = _run_archive_scan(tmp_path, data)
    assert status == 1
    assert "trailing gzip data" in output


@pytest.mark.parametrize("offset", [265, 297])
def test_tar_owner_group_values_are_intentional_format_metadata(tmp_path, offset):
    raw = _tar_bytes([("notes.txt", b"safe", tarfile.REGTYPE)], compressed=False)
    raw = _rewrite_tar_header(raw, offset, ARCHIVE_WORD.encode())
    status, output = _run_archive_scan(tmp_path, gzip.compress(raw, mtime=0))
    assert status == 0
    assert "scanned 1 text member(s)" in output


def test_archive_gzip_magic_is_recognized_without_extension(tmp_path):
    data = _tar_bytes([("notes.txt", f" {ARCHIVE_WORD} ".encode(), tarfile.REGTYPE)])
    status, output = _run_archive_scan(tmp_path, data, name="evidence.data")
    assert status == 1
    assert "prohibited word in archive member contents" in output
