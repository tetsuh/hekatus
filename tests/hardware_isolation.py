"""Process-wide pytest guard for accidental accelerator access.

The guard is deliberately implemented with Python audit events rather than
wrapping public subprocess or file APIs.  Audit events are emitted by the
interpreter before process creation and file opening, so aliases cached by
production modules remain covered and ordinary test doubles remain
mockable.  The exception derives from ``BaseException`` so a production
``except Exception`` cannot turn a blocked operation into a passing test.
"""

from __future__ import annotations

import contextvars
import os
import posixpath
import shlex
import sys
from typing import Any


class HardwareAccessViolation(BaseException):
    """A forbidden host/device operation reached the test process."""


_BLOCKED_EXECUTABLES = frozenset(
    {
        "tt-smi",
        "docker",
        "run_in_container.sh",
        "tt-topology",
        "tt-flash",
    }
)
_NON_WRAPPER_EXECUTABLES = _BLOCKED_EXECUTABLES - {"run_in_container.sh"}
_PROCESS_EVENTS = frozenset(
    {
        "os.exec",
        "os.execve",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execvp",
        "os.execvpe",
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "os.popen",
        "subprocess.Popen",
    }
)
_DEVICE_ROOT = "/dev/tenstorrent"
_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "zsh", "ksh", "fish"})
_COMMAND_PREFIXES = frozenset(
    {"env", "sudo", "command", "timeout", "stdbuf", "nice", "nohup"}
)

# ``None`` denies all guarded operations.  ``wrapper`` permits only the
# wrapper executable for explicitly opted-in wrapper tests; ``hardware``
# permits both process launch and device opens.  These values are set only by
# fixtures that require an explicit command-line opt-in.
_permission_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hekatus_test_hardware_permission", default=None
)
_audit_hook_installed = False
_original_os_open: Any = None


def install_audit_hook() -> None:
    """Install the deny-by-default hook once for this pytest interpreter."""
    global _audit_hook_installed
    if _audit_hook_installed:
        return
    sys.addaudithook(_audit_event)
    _install_os_open_boundary()
    _audit_hook_installed = True


def _install_os_open_boundary() -> None:
    global _original_os_open
    if _original_os_open is None:
        _original_os_open = os.open
        os.open = _guarded_os_open


def _guarded_os_open(
    path: Any,
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
) -> int:
    if (
        dir_fd is not None
        and not _path_is_absolute(path)
        and _permission_scope.get() != "hardware"
    ):
        base_dir = _dir_fd_target(dir_fd)
        if base_dir is None:
            raise HardwareAccessViolation("unclassifiable relative device open")
        if _is_device_path(path, base_dir=base_dir):
            raise HardwareAccessViolation("blocked device open")
    kwargs = {} if dir_fd is None else {"dir_fd": dir_fd}
    return _original_os_open(path, flags, mode, **kwargs)


def _path_is_absolute(value: Any) -> bool:
    try:
        candidate = os.fspath(value)
    except TypeError:
        return False
    if isinstance(candidate, bytes):
        candidate = os.fsdecode(candidate)
    return isinstance(candidate, str) and posixpath.isabs(candidate.replace("\\", "/"))


def _dir_fd_target(dir_fd: int) -> str | None:
    try:
        return os.readlink(f"/proc/self/fd/{int(dir_fd)}")
    except (OSError, TypeError, ValueError):
        return None


def deny_access() -> contextvars.Token[str | None]:
    """Deny guarded operations for the current test context."""
    return _permission_scope.set(None)


def restore_access(token: contextvars.Token[str | None]) -> None:
    """Restore the permission scope that preceded a fixture."""
    _permission_scope.reset(token)


def permit_hardware() -> contextvars.Token[str | None]:
    """Permit hardware operations after the explicit operator opt-in."""
    return _permission_scope.set("hardware")


def permit_wrapper() -> contextvars.Token[str | None]:
    """Permit only wrapper process launches after explicit operator opt-in."""
    return _permission_scope.set("wrapper")


def _audit_event(event: str, args: tuple[Any, ...]) -> None:
    scope = _permission_scope.get()
    if event == "open":
        if scope != "hardware" and args and _is_device_path(args[0]):
            raise HardwareAccessViolation("blocked device open")
        return

    if event not in _PROCESS_EVENTS:
        return
    if scope == "hardware":
        return
    if (
        scope == "wrapper"
        and _contains_wrapper_command(event, args)
        and not _contains_blocked_executable(event, args, _NON_WRAPPER_EXECUTABLES)
    ):
        return
    if _contains_blocked_executable(event, args):
        raise HardwareAccessViolation("blocked hardware process")


def _contains_wrapper_command(event: str, args: tuple[Any, ...]) -> bool:
    if event in {"os.system", "os.popen"}:
        return _shell_text_contains_name(args[0] if args else None)
    if event == "subprocess.Popen" or event == "os.posix_spawn":
        executable = args[0] if args else None
        argv = args[1] if len(args) > 1 else None
    elif event == "os.spawn":
        executable = args[1] if len(args) > 1 else None
        argv = args[2] if len(args) > 2 else None
    else:
        executable = args[0] if args else None
        argv = args[1] if len(args) > 1 else None
    return _named_token(executable) or _wrapper_argv(argv)


def _wrapper_argv(argv: Any) -> bool:
    if isinstance(argv, (str, bytes, os.PathLike)):
        return _shell_text_contains_name(argv)
    if not isinstance(argv, (tuple, list)) or not argv:
        return False
    tokens = [_decode_pathlike(token) for token in argv]
    first = _executable_name(tokens[0])
    if first == "run_in_container.sh":
        return True
    if first in _SHELL_EXECUTABLES:
        return _wrapper_shell_argv(tokens)
    return False


def _wrapper_shell_argv(tokens: list[str]) -> bool:
    skip_next = False
    for index, token in enumerate(tokens[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if token in {"-o", "--option"}:
            skip_next = True
            continue
        if token == "--":
            return index + 1 < len(tokens) and _named_token(tokens[index + 1])
        if token.startswith("-"):
            if "c" in token[1:]:
                payload = tokens[index + 1] if index + 1 < len(tokens) else ""
                return _shell_text_contains_name(payload)
            continue
        return _named_token(token)
    return False


def _shell_text_contains_name(value: Any) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return any(_named_token(token) for token in _shell_tokens(value))


def _named_token(value: Any) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return _executable_name(_decode_pathlike(value)) == "run_in_container.sh"


def _contains_blocked_executable(
    event: str,
    args: tuple[Any, ...],
    names: frozenset[str] = _BLOCKED_EXECUTABLES,
) -> bool:
    if event == "subprocess.Popen":
        executable = args[0] if args else None
        argv = args[1] if len(args) > 1 else None
        return _blocked_token(executable, names) or _blocked_argv(argv, names)
    if event in {"os.posix_spawn", "os.spawn"}:
        if event == "os.posix_spawn":
            executable = args[0] if args else None
            argv = args[1] if len(args) > 1 else None
        else:
            executable = args[1] if len(args) > 1 else None
            argv = args[2] if len(args) > 2 else None
        return _blocked_token(executable, names) or _blocked_argv(argv, names)
    if event.startswith("os.exec"):
        executable = args[0] if args else None
        argv = args[1] if len(args) > 1 else None
        return _blocked_token(executable, names) or _blocked_argv(argv, names)
    return _shell_text_is_blocked(args[0] if args else None, names)


def _blocked_argv(argv: Any, names: frozenset[str]) -> bool:
    if isinstance(argv, (str, bytes, os.PathLike)):
        return _shell_text_is_blocked(argv, names)
    if not isinstance(argv, (tuple, list)) or not argv:
        return False
    tokens = [_decode_pathlike(token) for token in argv]
    first = _executable_name(tokens[0])
    if first in names:
        return True
    if first in _SHELL_EXECUTABLES:
        return _blocked_shell_argv(tokens, names)
    if first in _COMMAND_PREFIXES:
        return any(_blocked_token(token, names) for token in tokens[1:])
    return False


def _blocked_shell_argv(tokens: list[str], names: frozenset[str]) -> bool:
    skip_next = False
    for index, token in enumerate(tokens[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if token in {"-o", "--option"}:
            skip_next = True
            continue
        if token == "--":
            return index + 1 < len(tokens) and _blocked_token(tokens[index + 1], names)
        if token.startswith("-"):
            if "c" in token[1:]:
                payload = tokens[index + 1] if index + 1 < len(tokens) else ""
                return _shell_text_is_blocked(payload, names)
            continue
        return _blocked_token(token, names)
    return False


def _shell_text_is_blocked(value: Any, names: frozenset[str]) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    tokens = _shell_tokens(value)
    return any(_blocked_token(token, names) for token in tokens)


def _shell_tokens(value: Any) -> list[str]:
    text = _decode_pathlike(value)
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return text.split()


def _blocked_token(value: Any, names: frozenset[str]) -> bool:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    return _executable_name(_decode_pathlike(value)) in names


def _decode_pathlike(value: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> str:
    try:
        value = os.fspath(value)
    except TypeError:
        return str(value)
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return value


def _executable_name(value: str) -> str:
    # Audit payloads may contain shell assignments or surrounding whitespace;
    # exact basename matching avoids blocking ordinary names that merely have
    # a guarded token as a substring.
    candidate = value.strip().replace("\\", "/").rstrip("/")
    return posixpath.basename(candidate)


def _is_device_path(value: Any, *, base_dir: str | None = None) -> bool:
    try:
        candidate = os.fspath(value)
    except TypeError:
        return False
    if isinstance(candidate, bytes):
        candidate = os.fsdecode(candidate)
    if not isinstance(candidate, str):
        return False
    candidate = candidate.replace("\\", "/")
    if not posixpath.isabs(candidate):
        candidate = posixpath.join(base_dir or os.getcwd(), candidate)
    # ``realpath`` resolves aliases through metadata only; it does not open the
    # target.  Resolve before comparing so a symlink outside /dev is covered.
    normalized = os.path.realpath(os.path.abspath(candidate))
    normalized = posixpath.normpath(normalized.replace("\\", "/"))
    normalized = "/" + normalized.lstrip("/")
    return normalized == _DEVICE_ROOT or normalized.startswith(_DEVICE_ROOT + "/")
