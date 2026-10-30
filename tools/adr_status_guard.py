"""Reject unsettled ADR header statuses in the checked-out tree (no Git needed)."""

import argparse
import re
from datetime import date
from pathlib import Path

STATUS_DECLARATION = re.compile(r"^\s*-\s*Status\s*:")
STATUS_LINE = re.compile(r"- Status: \*\*(.+)\*\* — ([0-9]{4}-[0-9]{2}-[0-9]{2})")
SETTLED_STATES = {"Accepted", "Rejected", "Deprecated"}
SUPERSEDED = re.compile(r"Superseded by ADR-[0-9]{4}")


def status_error(text: str) -> str | None:
    """Validate one declaration, anywhere in the file, then its header position."""
    lines = text.splitlines()
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
