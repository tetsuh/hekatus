"""Check only the three ADR-0006 traceability conditions in base..head.

Merge commits (>1 parent) and authors ending in [bot] are exempt from every
condition. Authored headers must have the ADR-0003 form, <=100 characters and
one trailing (#N); that issue needs a closing or Refs reference in the PR body.
Every closing issue needs an authored commit. Repository convention is Closes /
Refs; all nine GitHub closing keywords are recognized case-insensitively.

Body references are textual keyword + whitespace + #N matches, not a Markdown
or GitHub-link parser. Commit-body bullets, issue existence/API calls, commits
already on base, semantic English/imperative checks, Markdown rendering/negation,
full URLs and cross-repository references are out of scope. No additional
malformed-candidate heuristics are applied.
"""

import argparse
import re
import subprocess
from dataclasses import dataclass

HEADER = re.compile(
    r"(?:feat|fix|docs|test|refactor|build|ci|chore)\([^()\s]+\): "
    r"\S.* \(#\d+\)",
    re.ASCII,
)
TRAILING_REFS = re.compile(r"(?: \(#\d+\))+$", re.ASCII)
BODY_REF = re.compile(
    r"\b(close[sd]?|fix(?:es|ed)?|resolve[sd]?|refs)\s+#([0-9]+)\b",
    re.IGNORECASE | re.ASCII,
)
LOG_FORMAT = "%H%x00%P%x00%an%x00%s"


@dataclass(frozen=True)
class Commit:
    sha: str
    parents: tuple[str, ...]
    author: str
    header: str

    @property
    def exempt(self) -> bool:
        return len(self.parents) > 1 or self.author.endswith("[bot]")

    @property
    def label(self) -> str:
        # repr keeps untrusted headers on one inert diagnostic line.
        return f"{self.sha[:8]} {self.header!r}"


def parse_commits(text: str) -> list[Commit]:
    """Parse Git's four NUL-separated fields and NUL-terminated records."""
    if not text:
        return []
    fields = text.split("\0")
    if fields.pop() != "" or len(fields) % 4:
        raise ValueError("invalid Git commit records")
    return [
        Commit(sha, tuple(parents.split()), author, header)
        for sha, parents, author, header in zip(*[iter(fields)] * 4, strict=True)
    ]


def trailing_issue(header: str) -> str | None:
    """Only one token in the terminal reference run identifies an issue."""
    match = TRAILING_REFS.search(header)
    if match is None or match.group().count("(#") != 1:
        return None
    return match.group()[3:-1]


def violations(commits: list[Commit], body: str, *, head: Commit) -> list[str]:
    """Return one line per violation; uncovered closing issues use head context."""
    closing, referenced = set(), set()
    for match in BODY_REF.finditer(body):
        keyword, issue = match.groups()
        referenced.add(issue)
        if keyword.lower() != "refs":
            closing.add(issue)

    errors = []
    authored_issues = set()
    for commit in commits:
        if commit.exempt:
            continue
        issue = trailing_issue(commit.header)
        if HEADER.fullmatch(commit.header) is None or issue is None or len(commit.header) > 100:
            errors.append(
                f"traceability guard: {commit.label}: condition 1: expected "
                "<type>(<scope>): <summary> (#N), exactly one trailing reference, <=100 characters"
            )
        if issue is not None:
            authored_issues.add(issue)
            if issue not in referenced:
                errors.append(
                    f"traceability guard: {commit.label}: condition 2: "
                    f"issue #{issue} is absent from closing/Refs references in PR body"
                )
    for issue in sorted(closing - authored_issues):
        errors.append(
            f"traceability guard: {head.label}: condition 3: closing issue #{issue} "
            "has no authored commit (PR head context)"
        )
    return errors


def git_commits(revision: str, *, head_only: bool = False) -> list[Commit]:
    # CLI revisions are data, never Git options; retain local ref/range syntax.
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./~^@{}-]*", revision) is None:
        raise ValueError("invalid Git revision characters")
    command = ["git", "log", "-z", f"--format={LOG_FORMAT}"]
    if head_only:
        command.append("-1")
    command += ["--end-of-options", revision, "--"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return parse_commits(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--body-file", type=argparse.FileType("r", encoding="utf-8"), required=True)
    try:
        args = parser.parse_args()
        with args.body_file as stream:
            body = stream.read()
        commits = git_commits(f"{args.base}..{args.head}")
        head = git_commits(args.head, head_only=True)[0]
        errors = violations(commits, body, head=head)
    except (argparse.ArgumentError, OSError, ValueError, IndexError, subprocess.CalledProcessError):
        print("traceability guard: unable to read body or Git commit records")
        return 1
    if errors:
        print("\n".join(errors))
        return 1
    print("traceability guard: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
