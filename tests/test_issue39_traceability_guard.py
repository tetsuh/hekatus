"""Board-free coverage of the frozen Issue #39 invariants."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import traceability_guard as guard

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools/traceability_guard.py"
HEADER = "ci(traceability): check the conditions (#39)"


def record(header=HEADER, *, author="Fixture Author", parents="b" * 40, sha="a" * 40):
    """Exercise parsing rather than constructing records behind its boundary."""
    return guard.parse_commits(f"{sha}\0{parents}\0{author}\0{header}\0")[0]


def violations(commits, body="Closes #39", *, head=None):
    return guard.violations(commits, body, head=head or commits[0])


def test_record_fields_and_multiple_records():
    raw = "\0".join(("a" * 40, "b" * 40, "Fixture Author", HEADER)) + "\0"
    records = guard.parse_commits(raw + raw.replace("Fixture Author", "dependency[bot]"))
    assert len(records) == 2
    assert records[0].sha == "a" * 40
    assert records[0].parents == ("b" * 40,)
    assert records[0].author == "Fixture Author"
    assert records[0].header == HEADER
    assert records[1].author == "dependency[bot]"
    assert guard.parse_commits("") == []
    assert record(parents="").parents == ()


@pytest.mark.parametrize("kind", ["feat", "fix", "docs", "test", "refactor", "build", "ci", "chore"])
def test_header_types_pass(kind):
    assert violations([record(f"{kind}(component): describe the change (#39)")]) == []


@pytest.mark.parametrize(
    "header",
    [
        "describe the change",
        "ci(traceability): check the conditions",
        "ci: check the conditions (#39)",
        "ci(): check the conditions (#39)",
        "unknown(traceability): check the conditions (#39)",
        "ci(traceability):  (#39)",
        "ci(traceability): check (#39) (#40)",
        "ci(traceability): check (#39) (#39)",
        "ci(traceability): check (#３９)",
        "ci(traceability): check (#39) extra",
    ],
)
def test_condition1_rejects_invalid_headers(header):
    commit = record(header)
    errors = violations([commit], "Refs #39\nRefs #40")
    assert len(errors) == 1
    assert commit.sha[:8] in errors[0]
    assert repr(header) in errors[0]
    assert "condition 1" in errors[0]


@pytest.mark.parametrize("width", [100, 101])
def test_header_width_boundary(width):
    header = "ci(traceability): " + "x" * (width - len("ci(traceability):  (#39)")) + " (#39)"
    assert len(header) == width
    errors = violations([record(header)])
    assert bool(errors) == (width == 101)
    if errors:
        assert "condition 1" in errors[0] and "100" in errors[0]


def test_issue_prose_is_not_a_commit_reference():
    commit = record("fix(bench): redact Issue #146 diagnostics by allowlist (#150)")
    assert violations([commit], "Closes #150") == []
    errors = violations([commit], "Closes #150\nCloses #146")
    assert len(errors) == 1 and "condition 3" in errors[0] and "#146" in errors[0]


@pytest.mark.parametrize(
    "body",
    ["Closes #39", "Refs #39", "Closes #39\nRefs #40", "Refs #39\nRefs #40", "rEfS #39"],
)
def test_closes_refs_and_mixed_bodies_pass(body):
    assert violations([record()], body) == []


@pytest.mark.parametrize("keyword", ["close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved"])
@pytest.mark.parametrize("uppercase", [False, True])
def test_closing_keyword_variants_have_both_directions(keyword, uppercase):
    keyword = keyword.upper() if uppercase else keyword
    assert violations([record()], f"{keyword} #39") == []
    errors = violations([record()], f"Refs #39\n{keyword} #40")
    assert len(errors) == 1 and "condition 3" in errors[0] and "#40" in errors[0]


@pytest.mark.parametrize("body", ["", "Refs #40", "Closes #390", "Issue #39", "encloses #39"])
def test_condition2_requires_a_body_reference(body):
    errors = violations([record()], body)
    assert any("condition 2" in error and "#39" in error for error in errors)
    assert all("aaaaaaaa" in error and repr(HEADER) in error for error in errors)


def test_all_authored_issues_need_references_and_all_failures_are_reported():
    commits = [record(), record("test(traceability): cover another issue (#40)", sha="c" * 40)]
    errors = violations(commits, "")
    assert len(errors) == 2
    assert "aaaaaaaa" in errors[0] and "condition 2" in errors[0]
    assert "cccccccc" in errors[1] and "condition 2" in errors[1]
    assert violations(commits, "Closes #39\nRefs #40") == []


def test_condition3_is_distinct_closing_issues_only_with_head_context():
    head = record("Merge pull request #99", parents="b c", sha="d" * 40)
    errors = violations([record()], "Refs #39\nCloses #40\nCLOSED #40\nCloses #41", head=head)
    assert len(errors) == 2
    for error, issue in zip(errors, (40, 41), strict=True):
        assert "condition 3" in error and f"#{issue}" in error
        assert "dddddddd" in error and repr(head.header) in error
        assert "PR head context" in error
    assert violations([], "Refs #39", head=head) == []
    assert "condition 3" in violations([], "Closes #39", head=head)[0]


@pytest.mark.parametrize(
    ("header", "author", "parents"),
    [
        ("Merge pull request #99 from owner/branch", "Fixture Author", "b c"),
        ("Merge branch main into feature", "Fixture Author", "b c"),
        ("Merge octopus", "Fixture Author", "b c d"),
        ("chore(deps): bump pillow from 12.3.0 to 13.0.0", "dependabot[bot]", "b"),
        ("x" * 101, "another[bot]", "b"),
    ],
)
def test_merge_and_bot_commits_are_exempt_from_all_conditions(header, author, parents):
    exempt = record(header, author=author, parents=parents)
    assert violations([exempt], "") == []
    assert violations([exempt, record()], "Closes #39") == []
    # An otherwise valid exempt header is still not authored coverage.
    exempt = record(author=author, parents=parents)
    errors = violations([exempt], "Closes #39")
    assert len(errors) == 1 and "condition 3" in errors[0]


@pytest.mark.parametrize("author", ["bot", "person [bot] extra", "person[BOT]"])
def test_only_the_exact_author_suffix_is_exempt(author):
    assert "condition 1" in violations([record("unscoped", author=author)], "")[0]


def test_cli_temporary_repository(tmp_path):
    env = os.environ.copy()
    env.update(
        GIT_AUTHOR_NAME="Fixture Author",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_NAME="Fixture Committer",
        GIT_COMMITTER_EMAIL="committer@example.invalid",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
    )

    def git(*args, author=None):
        run_env = dict(env, GIT_AUTHOR_NAME=author) if author else env
        return subprocess.run(
            ["git", *args], cwd=tmp_path, env=run_env, text=True, capture_output=True,
            check=True, timeout=10,
        ).stdout.strip()

    def commit(header, *, author=None):
        git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", header, author=author)
        return git("rev-parse", "HEAD")

    def run(base, head, body):
        body_file = tmp_path / "body.md"
        body_file.write_text(body, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--base", base, "--head", head,
             "--body-file", str(body_file)],
            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10, check=False,
        )

    git("init", "-b", "main")
    base = commit("pre-existing unscoped commit without issue")
    git("checkout", "-b", "feature")
    first = commit(HEADER)
    commit("chore(deps): bump pillow from 12.3.0 to 13.0.0", author="dependabot[bot]")
    git("checkout", "-b", "side", first)
    commit("test(traceability): cover a side branch (#40)")
    git("checkout", "feature")
    git("-c", "commit.gpgsign=false", "merge", "--no-ff", "side", "-m", "Merge side branch")
    head = git("rev-parse", "HEAD")
    clean = run(base, head, "Closes #39\nRefs #40\nRefs #900")
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert clean.stdout == "traceability guard: clean\n" and clean.stderr == ""
    red = run(base, head, "Refs #40\nCloses #99")
    assert red.returncode == 1
    assert "condition 2" in red.stdout and first[:8] in red.stdout and repr(HEADER) in red.stdout
    assert "condition 3" in red.stdout and head[:8] in red.stdout and "Merge side branch" in red.stdout
    assert len(red.stdout.splitlines()) == 2
    # No retroactive scan of base, including an empty range with closing-body context.
    assert run(head, head, "").returncode == 0
    empty = run(head, head, "Closes #39")
    assert empty.returncode == 1 and "PR head context" in empty.stdout
    assert head[:8] in empty.stdout
    assert run("missing-revision", head, "").returncode == 1
    invalid = commit("ci: missing scope (#39)")
    result = run(base, invalid, "Closes #39\nRefs #40")
    assert result.returncode == 1 and "condition 1" in result.stdout
    assert invalid[:8] in result.stdout


def test_workflow_contract():
    text = (ROOT / ".github/workflows/traceability-guard.yml").read_text(encoding="utf-8")
    for required in (
        "name: traceability-guard", "  traceability-guard:",
        "types: [opened, synchronize, reopened, edited]", "contents: read",
        "actions/checkout@v7", "fetch-depth: 0", "persist-credentials: false",
        "ref: ${{ github.event.pull_request.head.sha }}",
        "PR_BODY: ${{ github.event.pull_request.body }}",
        'printf \'%s\' "$PR_BODY"', "python3 tools/traceability_guard.py",
        "BASE_SHA: ${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
    ):
        assert required in text
    assert "  push:" not in text and "${{ secrets." not in text
    run_text = text.split("run: |", 1)[1]
    assert "${{" not in run_text
