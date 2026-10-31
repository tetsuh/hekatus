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


@pytest.mark.parametrize("marker", ["```", "~~~", "````", "~~~~"])
@pytest.mark.parametrize("indent", range(4))
@pytest.mark.parametrize("info", ["", "text", "<!--"])
def test_fenced_examples_do_not_declare_status_or_start_sections(tmp_path, marker, indent, info):
    fence = " " * indent + marker
    example = f"{fence}{info}\n## Example\n{ACCEPTED}{fence}\n"
    path = write_adr(tmp_path, "0001-fixture.md", example)
    assert_failure(run_guard(tmp_path), path, "missing status line")
    path.write_text(f"# ADR\n{example}{ACCEPTED}\n## Context\n{example}")
    result = run_guard(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "example",
    [
        f"<!-- {ACCEPTED.rstrip()} -->\n",
        f"<!--\n## Example\n{ACCEPTED}-->\n",
        f"<!-- first --> <!--\n## Example\n{ACCEPTED}-->\n",
        f"<!--\n```text\n## Example\n{ACCEPTED}```\n-->\n",
        f"<!--\n~~~text\n## Example\n{ACCEPTED}~~~\n-->\n",
    ],
)
def test_commented_examples_do_not_declare_status_or_start_sections(tmp_path, example):
    path = write_adr(tmp_path, "0001-fixture.md", example)
    assert_failure(run_guard(tmp_path), path, "missing status line")
    path.write_text(f"# ADR\n{example}{ACCEPTED}\n## Context\n{example}")
    result = run_guard(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_comment_opened_in_fence_is_ignored(tmp_path, marker):
    path = write_adr(tmp_path, "0001-fixture.md", f"{marker}\n<!--\n{marker}\n{ACCEPTED}")
    assert run_guard(tmp_path).returncode == 0
    path.write_text(f"# ADR\n{marker}\n<!--\n{marker}\n## Real section\n{ACCEPTED}")
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


@pytest.mark.parametrize("opening", ["```", "~~~", "<!--"])
def test_unterminated_embedded_context_extends_to_eof(tmp_path, opening):
    example = f"{opening}\n## Example\n{ACCEPTED}"
    path = write_adr(tmp_path, "0001-fixture.md", example)
    assert_failure(run_guard(tmp_path), path, "missing status line")
    path.write_text(f"# ADR\n{ACCEPTED}{example}")
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("marker", ["```", "~~~"])
@pytest.mark.parametrize("false_close", ["short", "other", "info", "indented"])
def test_only_matching_sufficient_bare_fence_closes(tmp_path, marker, false_close):
    closer = {
        "short": marker[:2],
        "other": "~~~" if marker == "```" else "```",
        "info": marker + "text",
        "indented": "    " + marker,
    }[false_close]
    example = f"{marker}\n{closer}\n## Example\n{ACCEPTED}  {marker}x\n"
    # The final info-bearing line is not a closer either.
    path = write_adr(tmp_path, "0001-fixture.md", example)
    assert_failure(run_guard(tmp_path), path, "missing status line")
    path.write_text(f"# ADR\n{example}{marker}\t \n{ACCEPTED}")
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_longer_closing_fence_resumes_real_declaration_checks(tmp_path, marker):
    example = f"{marker}text\n{ACCEPTED}{marker + marker[0]}\n"
    path = write_adr(tmp_path, "0001-fixture.md", ACCEPTED + example + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "multiple status lines")
    path.write_text(f"# ADR\n{example}## Real section\n{ACCEPTED}")
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


@pytest.mark.parametrize("marker", ["```", "~~~"])
def test_four_space_fence_does_not_open_code_block(tmp_path, marker):
    write_adr(tmp_path, "0001-fixture.md", f"    {marker}\n{ACCEPTED}    {marker}\n")
    assert run_guard(tmp_path).returncode == 0


def test_four_space_indented_status_and_section_keep_existing_behavior(tmp_path):
    path = write_adr(tmp_path, "0001-fixture.md", "    " + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "malformed status line")
    path.write_text("# ADR\n    ## Indented section\n" + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


def test_backtick_in_info_string_does_not_open_fence(tmp_path):
    write_adr(tmp_path, "0001-fixture.md", "```bad`info\n" + ACCEPTED)
    assert run_guard(tmp_path).returncode == 0


def test_comments_do_not_join_physical_fragments_into_a_declaration(tmp_path):
    path = write_adr(tmp_path, "0001-fixture.md", "- Sta<!-- hidden -->tus: **Accepted** — 2026-08-12\n")
    assert_failure(run_guard(tmp_path), path, "missing status line")


def test_comment_closing_line_preserves_visible_section(tmp_path):
    path = tmp_path / "0001-fixture.md"
    path.write_text("# ADR\n<!--\n-->## Visible section\n" + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


def test_real_duplicate_after_comment_is_not_hidden(tmp_path):
    path = write_adr(tmp_path, "0001-fixture.md", ACCEPTED + "<!--\n" + ACCEPTED + "-->\n" + ACCEPTED)
    assert_failure(run_guard(tmp_path), path, "multiple status lines")


@pytest.mark.parametrize(
    "candidate",
    [
        "- Status **Proposed** — 2026-08-12",
        "- Status Proposed — 2026-08-12",
        "- Status = **Proposed** — 2026-08-12",
        "* Status: **Proposed** — 2026-08-12",
        "+ Status: **Proposed** — 2026-08-12",
        "- status: **Proposed** — 2026-08-12",
        "- STATUS: **Proposed** — 2026-08-12",
        "- **Status:** **Proposed** — 2026-08-12",
        "- Status： **Proposed** — 2026-08-12",
        "Status: **Proposed** — 2026-08-12",
        "- Status Unknown — 2026-08-12",
    ],
)
def test_malformed_status_candidates_cannot_hide_beside_accepted(tmp_path, candidate):
    path = write_adr(tmp_path, "0001-fixture.md", candidate + "\n")
    assert_failure(run_guard(tmp_path), path, "malformed status line")
    path.write_text(f"# ADR\n{ACCEPTED}{candidate}\n")
    assert_failure(run_guard(tmp_path), path, "multiple status lines")
    for wrapper in ("```\n{}\n```\n", "~~~\n{}\n~~~\n", "<!--\n{}\n-->\n"):
        path.write_text(f"# ADR\n{ACCEPTED}" + wrapper.format(candidate))
        assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize(
    "prose",
    [
        "Status history records the transition.",
        "- Status history: Accepted.",
        "Status updates are recorded here.",
        "- Status updates are recorded here.",
        "**Status** updates are recorded here.",
    ],
)
def test_status_history_prose_is_not_a_declaration(tmp_path, prose):
    write_adr(tmp_path, "0001-fixture.md", ACCEPTED + prose + "\n")
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("section", ["##", "  ##", "Context\n-------", "Context\n   ------- \t", "Context\n-"])
def test_empty_atx_and_setext_sections_end_header(tmp_path, section):
    path = tmp_path / "0001-fixture.md"
    path.write_text(f"# ADR\n\n{section}\n\n{ACCEPTED}")
    assert_failure(run_guard(tmp_path), path, "status line must be in header")
    for wrapper in ("```\n{}\n```\n", "~~~\n{}\n~~~\n", "<!--\n{}\n-->\n"):
        path.write_text("# ADR\n\n" + wrapper.format(section) + ACCEPTED)
        assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize(
    "prefix",
    ["\n", "# ADR\n", "- Item\n", "1. Item\n", "123) Item\n", "> Quote\n", "Paragraph\n\n", "Paragraph\n<!-- hidden -->\n", "Paragraph\n```\n```\n"],
)
def test_thematic_break_without_adjacent_paragraph_does_not_end_header(tmp_path, prefix):
    path = tmp_path / "0001-fixture.md"
    path.write_text(prefix + "---\n\n" + ACCEPTED)
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize(
    "literal",
    [
        "The opener is `<!--`.",
        "The opener is ``<!--``.",
        "The opener is `` ` <!-- ` ``.",
        r"The opener is \<!--.",
        '<span title="<!--">marker</span>',
        "<span title='<!--'>marker</span>",
        "The opener is `across\n<!-- lines`.",
        '<span title="across\n<!-- lines">marker</span>',
    ],
)
@pytest.mark.parametrize(
    "status",
    [
        "- Status: **Proposed** — 2026-08-12\n",
        "- Status: **Unknown** — 2026-08-12\n",
        "- Status: Proposed — 2026-08-12\n",
        ACCEPTED,
    ],
)
def test_literal_comment_openers_do_not_hide_real_statuses(tmp_path, literal, status):
    path = write_adr(tmp_path, "0001-fixture.md", ACCEPTED + literal + "\n" + status)
    assert_failure(run_guard(tmp_path), path, "multiple status lines")
    path.write_text(f"# ADR\n{literal}\n## Context\n-->\n{ACCEPTED}")
    assert_failure(run_guard(tmp_path), path, "status line must be in header")


@pytest.mark.parametrize("literal", ["`<!--`", "``<!--``", r"\<!--", '<span title="<!--">x</span>'])
def test_real_comment_after_literal_opener_still_masks_examples(tmp_path, literal):
    write_adr(tmp_path, "0001-fixture.md", ACCEPTED + literal + "\n<!--\n" + ACCEPTED + "-->\n")
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("literal", ["`", "``", '<span title="'])
def test_literal_looking_text_inside_real_comment_does_not_hide_closer(tmp_path, literal):
    write_adr(tmp_path, "0001-fixture.md", f"<!-- {literal} -->\n{ACCEPTED}")
    assert run_guard(tmp_path).returncode == 0


@pytest.mark.parametrize("prefix", ["Unmatched ` text ", "Unmatched `` text ", r"Escaped backtick \` ", r"Even slashes \\"])
def test_unmatched_or_escaped_code_marker_does_not_hide_real_comment(tmp_path, prefix):
    path = write_adr(tmp_path, "0001-fixture.md", prefix + "<!--\n" + ACCEPTED + "-->\n")
    assert_failure(run_guard(tmp_path), path, "missing status line")


def test_inline_code_lookahead_does_not_cross_fence_or_paragraph(tmp_path):
    for boundary in ("\n\n", "\n~~~\n~~~\n"):
        path = write_adr(tmp_path, "0001-fixture.md", "Unmatched `<!--" + boundary + "`\n" + ACCEPTED)
        assert_failure(run_guard(tmp_path), path, "missing status line")


def test_repository_adrs_pass():
    result = run_guard()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "adr status guard: clean\n"
