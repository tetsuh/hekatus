from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import newton_schulz_bringup as bringup

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPUTE_KERNEL_DIRECTORY = REPOSITORY_ROOT / "enodia/tt/bench/kernels"
MATMUL_FAMILY_NAMES = frozenset(
    {
        "matmul_tiles",
        "matmul_block",
        "matmul_init",
        "matmul_block_init",
        "matmul_one",
        "matmul_group",
    }
)
_CALL_PATTERN = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*(?:<[^()\n;{}]*>\s*)?\(")
_INIT_COMMON_PATTERN = re.compile(r"\b(?P<name>[A-Za-z_]\w*_init_common)\s*\(")
_DEFINITION_SUFFIX = re.compile(
    r"\s*(?:(?:const|volatile|override|final)\b|"
    r"noexcept(?:\s*\([^)]*\))?|requires\b[^{};]*|->[^{};]+)*\s*\{"
)
_DECLARATION_PREFIX = re.compile(
    r"(?:^|[;{}])\s*(?:(?:inline|static|constexpr|extern|template)\s+)*"
    r"(?:void|auto|bool|char|short|int|long|float|double|"
    r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*$"
)


@dataclass(frozen=True)
class _CallSite:
    name: str
    offset: int
    line: int


@dataclass(frozen=True)
class _OrderingViolation:
    first_matmul: _CallSite
    later_init_common: _CallSite


@dataclass(frozen=True)
class _SourceRange:
    opening: int
    closing: int


# These are historical diagnostics and the legacy scaffold.  Their full binary
# helpers intentionally remain after a matmul helper body; the exclusions keep
# this source rule focused while requiring every exception to stay observable.
INIT_COMMON_ORDERING_EXCLUSIONS = {
    "enodia/tt/bench/kernels/bringup_complex_compute.cpp": (
        "Legacy Stage 4 diagnostic keeps its binary helper after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_complex_modern_compute.cpp": (
        "Legacy Stage 41 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_complex_two_groups_compute.cpp": (
        "Legacy Stage 42 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_one_compute_copy_compute.cpp": (
        "Legacy Stage 45 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_one_correct_reader_copy_compute.cpp": (
        "Legacy Stage 48 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_residual_compute.cpp": (
        "Legacy Stage 43 diagnostic keeps its binary helper after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_residual_copy_compute.cpp": (
        "Legacy Stage 44 diagnostic keeps its binary helper after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_residual_correct_compute.cpp": (
        "Legacy Stage 46 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_newton_residual_correct_reader_copy_compute.cpp": (
        "Legacy Stage 47 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_eight_compute.cpp": (
        "Legacy Stage 6 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_first_residual_compute.cpp": (
        "Legacy Stage 68 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_first_residual_float32_boundary_compute.cpp": (
        "Legacy Stage 71 variant keeps its full binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_first_residual_waypoint_compute.cpp": (
        "Legacy Stage 69 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_float32_state_compute.cpp": (
        "Legacy Stage 61 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_four_plus_four_compute.cpp": (
        "Legacy Stage 62 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_four_plus_four_distinct_output_compute.cpp": (
        "Legacy Stage 67 diagnostic keeps its binary helpers after matmul."
    ),
    "enodia/tt/bench/kernels/bringup_ns_one_compute.cpp": (
        "Legacy Stage 5 diagnostic keeps its binary helpers after matmul."
    ),
}

# Keep additions to the exception list deliberate rather than allowing a new
# violating source to become invisible merely by adding one dictionary entry.
_EXPECTED_INIT_COMMON_ORDERING_EXCLUSIONS = frozenset(
    {
        "enodia/tt/bench/kernels/bringup_complex_compute.cpp",
        "enodia/tt/bench/kernels/bringup_complex_modern_compute.cpp",
        "enodia/tt/bench/kernels/bringup_complex_two_groups_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_one_compute_copy_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_one_correct_reader_copy_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_residual_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_residual_copy_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_residual_correct_compute.cpp",
        "enodia/tt/bench/kernels/bringup_newton_residual_correct_reader_copy_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_eight_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_first_residual_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_first_residual_float32_boundary_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_first_residual_waypoint_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_float32_state_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_four_plus_four_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_four_plus_four_distinct_output_compute.cpp",
        "enodia/tt/bench/kernels/bringup_ns_one_compute.cpp",
    }
)


def _mask_non_code(source: str) -> str:
    """Replace comments and literals while preserving source line positions."""
    masked = list(source)
    state = "code"
    index = 0
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if character == "/" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "line_comment"
            elif character == "/" and following == "*":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "block_comment"
            elif character == '"':
                masked[index] = " "
                index += 1
                state = "double_quote"
            elif character == "'":
                masked[index] = " "
                index += 1
                state = "single_quote"
            else:
                index += 1
        elif state == "line_comment":
            if character == "\n":
                state = "code"
            else:
                masked[index] = " "
            index += 1
        elif state == "block_comment":
            if character == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "code"
            else:
                if character != "\n":
                    masked[index] = " "
                index += 1
        else:
            if character == "\\":
                masked[index] = " "
                if index + 1 < len(source):
                    if source[index + 1] != "\n":
                        masked[index + 1] = " "
                    index += 2
                else:
                    index += 1
            else:
                if character == "\n":
                    index += 1
                    continue
                masked[index] = " "
                index += 1
                if (state == "double_quote" and character == '"') or (
                    state == "single_quote" and character == "'"
                ):
                    state = "code"
    return "".join(masked)


def _matching_close_parenthesis(source: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _matching_close_brace(source: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _function_range(source: str, name: str) -> _SourceRange | None:
    pattern = re.compile(rf"\b{re.escape(name)}\s*\([^)]*\)\s*\{{")
    match = pattern.search(source)
    if match is None:
        return None
    opening = match.end() - 1
    closing = _matching_close_brace(source, opening)
    if closing is None:
        return None
    return _SourceRange(opening, closing)


def _is_definition(source: str, closing_parenthesis: int) -> bool:
    suffix = source[closing_parenthesis + 1 : closing_parenthesis + 256]
    return _DEFINITION_SUFFIX.match(suffix) is not None


def _is_declaration(source: str, name_start: int) -> bool:
    prefix = source[max(0, name_start - 160) : name_start]
    return _DECLARATION_PREFIX.search(prefix) is not None


def _call_sites(source: str, pattern: re.Pattern[str], names: frozenset[str] | None = None):
    for match in pattern.finditer(source):
        name = match.group("name")
        if names is not None and name not in names:
            continue
        closing_parenthesis = _matching_close_parenthesis(source, match.end() - 1)
        if closing_parenthesis is None:
            continue
        if _is_definition(source, closing_parenthesis) or _is_declaration(source, match.start()):
            continue
        yield _CallSite(
            name=name,
            offset=match.start(),
            line=source.count("\n", 0, match.start()) + 1,
        )


def _ordering_violations(source: str) -> tuple[_OrderingViolation, ...]:
    masked = _mask_non_code(source)
    matmul_calls = tuple(_call_sites(masked, _CALL_PATTERN, MATMUL_FAMILY_NAMES))
    if not matmul_calls:
        return ()
    first_matmul = min(matmul_calls, key=lambda call: call.offset)
    init_common_calls = tuple(_call_sites(masked, _INIT_COMMON_PATTERN))
    kernel_main = _function_range(masked, "kernel_main")
    kernel_matmul_calls = tuple(
        call
        for call in matmul_calls
        if kernel_main is not None
        and kernel_main.opening < call.offset < kernel_main.closing
    )
    first_kernel_matmul = (
        min(kernel_matmul_calls, key=lambda call: call.offset)
        if kernel_matmul_calls
        else None
    )

    def is_before_kernel_matmul(call: _CallSite) -> bool:
        return (
            kernel_main is not None
            and first_kernel_matmul is not None
            and kernel_main.opening < call.offset < kernel_main.closing
            and call.offset < first_kernel_matmul.offset
        )

    return tuple(
        _OrderingViolation(first_matmul, init_common)
        for init_common in init_common_calls
        if not is_before_kernel_matmul(init_common)
    )


def _source_key(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _compute_sources() -> tuple[Path, ...]:
    return tuple(sorted(COMPUTE_KERNEL_DIRECTORY.glob("*_compute.cpp")))


def _display_path(path: Path) -> str:
    try:
        return _source_key(path)
    except ValueError:
        return path.name


def _format_violations(path: Path, violations: tuple[_OrderingViolation, ...]) -> str:
    return "\n".join(
        f"{_display_path(path)}:{violation.later_init_common.line}: "
        f"{violation.later_init_common.name}() follows "
        f"{violation.first_matmul.name}() at line {violation.first_matmul.line}"
        for violation in violations
    )


def _assert_source_is_clean(path: Path) -> None:
    violations = _ordering_violations(path.read_text(encoding="utf-8"))
    assert not violations, _format_violations(path, violations)


def test_all_non_excluded_compute_sources_pass():
    sources = _compute_sources()
    assert sources
    for source in sources:
        if _source_key(source) in INIT_COMMON_ORDERING_EXCLUSIONS:
            continue
        _assert_source_is_clean(source)


def test_synthetic_late_init_common_is_rejected_with_path_and_line(tmp_path):
    source = tmp_path / "synthetic_compute.cpp"
    source.write_text(
        """
void matmul_one() {
    matmul_block(left, right, 0, 0, 0, false, 1, 1, 1);
}
void binary_op_init_common(int left, int right, int output) {}
void kernel_main() {
    // matmul_tiles(left, right, 0, 0, 0); binary_op_init_common(left, right, output);
    matmul_one();
    binary_op_init_common(left, right, output);
}
""",
        encoding="utf-8",
    )

    violations = _ordering_violations(source.read_text(encoding="utf-8"))
    assert len(violations) == 1
    assert violations[0].first_matmul.name == "matmul_block"
    assert violations[0].later_init_common.name == "binary_op_init_common"
    with pytest.raises(AssertionError, match=r"synthetic_compute\.cpp:\d+:"):
        _assert_source_is_clean(source)


def test_helper_defined_before_matmul_is_checked_in_runtime_order(tmp_path):
    source = tmp_path / "synthetic_helper_order.cpp"
    source.write_text(
        """
void binary_op_helper() {
    binary_op_init_common(left, right, output);
}
void matmul_one() {
    matmul_block(left, right, 0, 0, 0, false, 1, 1, 1);
}
void kernel_main() {
    binary_op_init_common(left, right, output);
    matmul_one();
    binary_op_helper();
}
""",
        encoding="utf-8",
    )

    violations = _ordering_violations(source.read_text(encoding="utf-8"))
    assert len(violations) == 1
    assert violations[0].first_matmul.name == "matmul_block"
    assert violations[0].later_init_common.name == "binary_op_init_common"
    with pytest.raises(AssertionError, match=r"synthetic_helper_order\.cpp:\d+:"):
        _assert_source_is_clean(source)


def test_legacy_exclusions_are_explicit_and_still_violating():
    assert frozenset(INIT_COMMON_ORDERING_EXCLUSIONS) == _EXPECTED_INIT_COMMON_ORDERING_EXCLUSIONS
    source_keys = {_source_key(source) for source in _compute_sources()}
    assert set(INIT_COMMON_ORDERING_EXCLUSIONS) <= source_keys
    for source_key, reason in INIT_COMMON_ORDERING_EXCLUSIONS.items():
        assert source_key.endswith("_compute.cpp")
        assert reason.strip()
        source = REPOSITORY_ROOT / source_key
        violations = _ordering_violations(source.read_text(encoding="utf-8"))
        assert violations, _format_violations(source, violations)


def test_stage_70_and_stage_72_compute_sources_are_scanned_and_clean():
    sources = {_source_key(source): source for source in _compute_sources()}
    for stage_number in (70, 72):
        source = COMPUTE_KERNEL_DIRECTORY / bringup.STAGES[stage_number].compute_source
        source_key = _source_key(source)
        assert source_key in sources
        assert source_key not in INIT_COMMON_ORDERING_EXCLUSIONS
        _assert_source_is_clean(source)
