"""Fixed, board-free asset preflight for the Issue #120 pinned image.

Inventories describe names, kinds, sizes and executable attributes, not file
content hashes. The immutable image digest and existing library SHA-256 remain
the content provenance boundary. No accelerator module is imported here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PINNED_RUNTIME_ROOT = Path("/opt/venv/lib/python3.12/site-packages/ttnn")


@dataclass(frozen=True)
class RuntimeAssetTree:
    name: str
    path: Path
    inventory_sha256: str


# Audited with stdlib-only reads in the exact pinned image, with no devices.
ASSET_TREES = (
    RuntimeAssetTree("metal", PINNED_RUNTIME_ROOT / "tt_metal",
                     "8d85e48cca4c004fa41962846378fd06bd28bf57321ce24d7a9cafaf27a16314"),
    RuntimeAssetTree("runtime", PINNED_RUNTIME_ROOT / "runtime",
                     "56fe4e441edf3f309dca55bd97db56a03c3f060e8bafa6bd1db2d3f05b41ac8b"),
    RuntimeAssetTree("libraries", PINNED_RUNTIME_ROOT / "build/lib",
                     "06f873632238323b3891239edaca92e54eebc7e674755d0749e14c533df9c8f2"),
    RuntimeAssetTree("sfpi", Path("/opt/tenstorrent/sfpi"),
                     "fb5e85084358f9deda5081a6ee953c5ee3360f7ce084da84b63a42a51b18ca8b"),
)


class RuntimeAssetError(ValueError):
    """The fixed runtime cannot be safely inspected or lacks pinned assets."""


def _require_directory(path: Path) -> None:
    if not path.is_dir() or path.resolve() != path.absolute():
        raise RuntimeAssetError("runtime directory is missing, wrong-kind or redirected")


def _asset_row(root: Path, path: Path) -> dict:
    if path.is_symlink():
        # None of the four pinned asset trees has symlinks, even internal ones.
        raise RuntimeAssetError("runtime asset symlinks are not part of the pinned layout")
    row = {"path": path.relative_to(root).as_posix()}
    if path.is_file():
        metadata = path.stat()
        row.update(kind="file", size=metadata.st_size, executable=bool(metadata.st_mode & 0o111))
    elif path.is_dir():
        row["kind"] = "directory"
    else:
        raise RuntimeAssetError("unsupported runtime asset type")
    return row


def _inventory_sha256(root: Path) -> str:
    rows = [_asset_row(root, path) for path in sorted(root.rglob("*"))]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_runtime_assets() -> Path:
    """Validate only program-owned paths; there is deliberately no root argument."""
    try:
        _require_directory(PINNED_RUNTIME_ROOT)
        for tree in ASSET_TREES:
            _require_directory(tree.path)
            if _inventory_sha256(tree.path) != tree.inventory_sha256:
                raise RuntimeAssetError(f"runtime asset inventory differs from the pinned image: {tree.name}")
    except (OSError, RuntimeError) as exc:
        raise RuntimeAssetError("runtime assets could not be inspected") from exc
    return PINNED_RUNTIME_ROOT
