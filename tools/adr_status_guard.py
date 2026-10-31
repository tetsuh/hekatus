"""Reject unsettled ADR header statuses in the checked-out tree (no Git needed)."""

import argparse
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

# Recognize malformed candidates too; only STATUS_LINE defines accepted syntax.
STATUS_DECLARATION = re.compile(
    r"^\s*(?:[-+*]\s*)?(\*\*)?Status\b(.*)", re.IGNORECASE | re.ASCII
)
# ASCII keeps dates and ADR numbers restricted to the documented digit spelling.
STATUS_LINE = re.compile(r"- Status: \*\*(.+)\*\* — (\d{4}-\d{2}-\d{2})", re.ASCII)
SETTLED_STATES = {"Accepted", "Rejected", "Deprecated"}
SUPERSEDED = re.compile(r"Superseded by ADR-\d{4}", re.ASCII)
FENCE_OPEN = re.compile(r" {0,3}(`{3,}+|~{3,}+)(.*)")


def code_span_end(text: str, position: int) -> int:
    """Find an exactly matching backtick run within this paragraph, if any."""
    marker = re.match(r"`+", text[position:]).group()
    start = position + len(marker)
    line_start = text.rfind("\n", 0, position) + 1
    if re.fullmatch(r" {4,}", text[line_start:position]):
        # Preserve the guard's explicitly retained indented-code behavior.
        return start
    boundary = re.search(r"\n[ \t]*\n|\n {0,3}(?:`{3,}+|~{3,}+)", text[start:])
    stop = len(text) if boundary is None else start + boundary.start()
    for run in re.finditer(r"`+", text[start:stop]):
        if len(run.group()) == len(marker):
            return start + run.end()
    # Consume the entire unmatched run, never reinterpret one of its suffixes.
    return start


def tag_end(text: str, position: int) -> int:
    """Skip a complete tag, respecting quoted attributes across physical lines."""
    prefix = re.match(r"</?[A-Za-z][A-Za-z\d-]*", text[position:], re.ASCII)
    if prefix is None:
        return position
    quote = None
    for index in range(position + prefix.end(), len(text)):
        character = text[index]
        if quote is not None:
            if character == quote:
                quote = None
        elif character in "\"'":
            quote = character
        elif character == ">":
            return index + 1
        elif character == "<":
            return position
    return position


def literal_end(text: str, position: int) -> int:
    """Literal spans cannot open HTML comments; unmatched markers remain text."""
    character = text[position]
    if character == "\\":
        return min(position + 2, len(text))
    if character == "`":
        return code_span_end(text, position)
    if character == "<":
        return tag_end(text, position)
    return position


def mask_comments(
    text: str, start: int, stop: int, in_comment: bool, protected_until: int
) -> tuple[str, bool, int]:
    """Mask comments/literals while preserving columns and physical boundaries."""
    parts = []
    position = start
    while position < stop:
        if in_comment:
            end = text.find("-->", position, stop)
            following = stop if end == -1 else end + 3
            parts.append(" " * (following - position))
            position = following
            in_comment = end == -1
        elif position < protected_until:
            following = min(stop, protected_until)
            parts.append(" " * (following - position))
            position = following
        elif text.startswith("<!--", position):
            in_comment = True
        else:
            protected_until = literal_end(text, position)
            if protected_until == position:
                parts.append(text[position])
                position += 1
    return "".join(parts), in_comment, protected_until


def declaration_lines(text: str) -> Iterator[str]:
    """Yield physical lines outside fences, with HTML comment contents masked."""
    closing_fence = None
    in_comment = False
    protected_until = 0
    lines = text.splitlines()
    text = "\n".join(lines)
    offset = 0
    for line in lines:
        start, stop = offset, offset + len(line)
        offset = stop + 1
        # Fence contents are literal: even an unclosed comment opener is inert.
        if closing_fence is not None:
            if closing_fence.fullmatch(line):
                closing_fence = None
            yield ""
            continue
        opening = None if in_comment or start < protected_until else FENCE_OPEN.fullmatch(line)
        if opening is not None:
            marker, info = opening.groups()
            if marker[0] == "~" or "`" not in info:
                closing_fence = re.compile(r" {0,3}" + re.escape(marker[0]) + rf"{{{len(marker)},}}[ \t]*")
                yield ""
                continue
        line, in_comment, protected_until = mask_comments(
            text, start, stop, in_comment, protected_until
        )
        yield line


def is_status_candidate(line: str) -> bool:
    """Distinguish malformed declaration-shaped values from ordinary status prose."""
    match = STATUS_DECLARATION.match(line)
    if match is None:
        return False
    bold_label, tail = match.groups()
    if bold_label and tail.startswith("**"):
        tail = tail[2:]
    value = tail.lstrip()
    state_word = value.split(maxsplit=1)[0].casefold() if value else ""
    return (
        value.startswith((":", "：", "=", "**"))
        or state_word in {"accepted", "rejected", "deprecated", "proposed", "superseded"}
        or " — " in tail
    )


def starts_section(lines: list[str], index: int) -> bool:
    """Recognize level-two ATX headings and dash underlines of paragraph text."""
    if re.match(r"^\s*##(?:\s|$)", lines[index]):
        return True
    if index == 0 or re.fullmatch(r" {0,3}-+[ \t]*", lines[index]) is None:
        return False
    previous = lines[index - 1]
    # An underline needs paragraph text, not a blank, heading, list, or quote.
    return bool(previous.strip()) and not re.match(
        r"^\s*(?:#{1,6}(?:\s|$)|[-+*](?:\s|$)|\d{1,9}[.)](?:\s|$)|>)", previous, re.ASCII
    )


def status_error(text: str) -> str | None:
    """Validate one declaration, anywhere in the file, then its header position."""
    lines = list(declaration_lines(text))
    declarations = [(index, line) for index, line in enumerate(lines) if is_status_candidate(line)]
    if not declarations:
        return "missing status line"
    if len(declarations) != 1:
        return "multiple status lines"
    index, line = declarations[0]
    match = STATUS_LINE.fullmatch(line)
    if match is None:
        return "malformed status line (expected - Status: **<state>** — YYYY-MM-DD)"
    if any(starts_section(lines, earlier) for earlier in range(index)):
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
