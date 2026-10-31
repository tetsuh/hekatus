"""Reject unsettled ADR header statuses in the checked-out tree (no Git needed)."""

import argparse
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

STATUS_DECLARATION = re.compile(r"^\s*-\s*Status\s*:")
# ASCII keeps dates and ADR numbers restricted to the documented digit spelling.
STATUS_LINE = re.compile(r"- Status: \*\*(.+)\*\* — (\d{4}-\d{2}-\d{2})", re.ASCII)
SETTLED_STATES = {"Accepted", "Rejected", "Deprecated"}
SUPERSEDED = re.compile(r"Superseded by ADR-\d{4}", re.ASCII)
FENCE_OPEN = re.compile(r" {0,3}(`{3,}+|~{3,}+)(.*)")


def mask_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Blank comment spans without joining physical fragments or shifting columns."""
    parts = []
    position = 0
    while position < len(line):
        if in_comment:
            end = line.find("-->", position)
            stop = len(line) if end == -1 else end + 3
            parts.append(" " * (stop - position))
            position = stop
            in_comment = end == -1
        else:
            start = line.find("<!--", position)
            if start == -1:
                parts.append(line[position:])
                break
            parts.append(line[position:start])
            position = start
            in_comment = True
    return "".join(parts), in_comment


def declaration_lines(text: str) -> Iterator[str]:
    """Yield physical lines outside fences, with HTML comment contents masked."""
    closing_fence = None
    in_comment = False
    for line in text.splitlines():
        # Fence contents are literal: even an unclosed comment opener is inert.
        if closing_fence is not None:
            if closing_fence.fullmatch(line):
                closing_fence = None
            continue
        opening = None if in_comment else FENCE_OPEN.fullmatch(line)
        if opening is not None:
            marker, info = opening.groups()
            if marker[0] == "~" or "`" not in info:
                closing_fence = re.compile(r" {0,3}" + re.escape(marker[0]) + rf"{{{len(marker)},}}[ \t]*")
                continue
        line, in_comment = mask_comments(line, in_comment)
        yield line


def status_error(text: str) -> str | None:
    """Validate one declaration, anywhere in the file, then its header position."""
    lines = list(declaration_lines(text))
    declarations = [(index, line) for index, line in enumerate(lines) if STATUS_DECLARATION.match(line)]
    if not declarations:
        return "missing status line"
    if len(declarations) != 1:
        return "multiple status lines"
    index, line = declarations[0]
    match = STATUS_LINE.fullmatch(line)
    if match is None:
        return "malformed status line (expected - Status: **<state>** — YYYY-MM-DD)"
    if any(re.match(r"^\s*##\s", earlier) for earlier in lines[:index]):
        return "status line must be in header before the first section"
    state, effective_date = match.groups()
    if state == "Proposed":
        return "Proposed is not settled; Proposed never reaches main"
    if state not in SETTLED_STATES and SUPERSEDED.fullmatch(state) is None:
        return f"unknown state: {state!r}"
    try:
        date.fromisoformat(effective_date)
    except ValueError:
        return "invalid status date"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/adr"), help="ADR directory to scan")
    root = parser.parse_args().root
    try:
        # iterdir reports missing/unreadable directories rather than glob's empty success.
        paths = sorted(path for path in root.iterdir() if path.match("*.md"))
        if not paths:
            raise ValueError("no Markdown ADRs found")
    except (OSError, ValueError) as error:
        print(f"{str(root)!r}: unable to scan ADR directory ({error})")
        return 1

    failed = False
    for path in paths:
        try:
            reason = status_error(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            reason = "unable to read ADR"
        if reason is not None:
            # repr keeps filenames on one inert line; do not emit workflow commands.
            print(f"{str(path)!r}: {reason}")
            failed = True
    if failed:
        return 1
    print("adr status guard: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
