"""Shared run-bound artifact path and identity checks for benchmark runners."""

from __future__ import annotations

import os
import re
from pathlib import Path

RUN_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
_RUN_ID_RE = re.compile(rf"^{RUN_ID_PATTERN}$")


class RunBindingError(ValueError):
    """An artifact path or embedded identity is not bound to the current run."""


def validate_run_id(value: object) -> str:
    """Return a safe run identifier suitable for deterministic artifact names."""
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise ValueError("run_id must be a safe artifact identifier")
    return value


def current_run_id() -> str:
    """Read and validate the wrapper-provided run identity."""
    value = os.environ.get("HEKATUS_TT_RUN_ID")
    try:
        return validate_run_id(value)
    except ValueError as exc:
        raise ValueError(
            "HEKATUS_TT_RUN_ID must be a nonempty safe filename component"
        ) from exc


def run_artifact_name(*, prefix: str, suffix: str, run_id: object) -> str:
    """Return the deterministic basename for one run-owned artifact."""
    return f"{prefix}{validate_run_id(run_id)}{suffix}"


def run_artifact_path(
    output_dir: Path, *, prefix: str, suffix: str, run_id: object
) -> Path:
    """Return the deterministic path without scanning the output directory."""
    return Path(output_dir) / run_artifact_name(prefix=prefix, suffix=suffix, run_id=run_id)


def require_run_artifact(
    output_dir: Path,
    *,
    prefix: str,
    suffix: str,
    run_id: object,
    explicit: Path | None = None,
    kind: str,
) -> Path:
    """Select only the exact current-run artifact and fail closed otherwise.

    ``explicit`` is accepted for callers that receive a path from a wrapper,
    but it must resolve to the same deterministic path as the omitted-path
    form.  There is deliberately no directory scan or most-recent fallback.
    """
    expected = run_artifact_path(output_dir, prefix=prefix, suffix=suffix, run_id=run_id)
    candidate = expected if explicit is None else Path(explicit)
    try:
        candidate_matches = candidate.name == expected.name and candidate.resolve() == expected.resolve()
    except (OSError, RuntimeError):
        candidate_matches = False
    if not candidate_matches or candidate.is_symlink():
        raise RunBindingError(
            f"{kind} path is not the exact current run artifact {expected.name}; "
            "glob fallback is disabled"
        )
    if not candidate.exists():
        raise FileNotFoundError(
            f"required {kind} artifact {expected.name} is absent; "
            "glob fallback is disabled"
        )
    if not candidate.is_file():
        raise RunBindingError(f"required {kind} artifact {expected.name} is not a regular file")
    return candidate


__all__ = [
    "RUN_ID_PATTERN",
    "RunBindingError",
    "current_run_id",
    "require_run_artifact",
    "run_artifact_name",
    "run_artifact_path",
    "validate_run_id",
]
