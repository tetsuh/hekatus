"""Process-wide pytest guard for accidental accelerator access.

The guard combines Python audit events with a narrow ``os.open`` boundary for
relative ``dir_fd`` paths. Audit events are emitted by the interpreter before
process creation and file opening, while the boundary supplies the directory
metadata unavailable in that event. Ordinary test doubles remain mockable.
The exception derives from ``BaseException`` so a production ``except
Exception`` cannot turn a blocked operation into a passing test.
"""

from __future__ import annotations

import contextvars
import os
import posixpath
import shlex
import stat
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
        "os.posix_spawnp",
        "os.spawn",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
        "os.popen",
        "subprocess.Popen",
    }
)
_POSIX_SPAWN_EVENTS = frozenset({"os.posix_spawn", "os.posix_spawnp"})
_SPAWN_EVENTS = frozenset(
    {"os.spawn", "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe"}
)
_EXEC_ENV_EVENTS = frozenset({"os.execve", "os.execle", "os.execlpe", "os.execvpe"})
_WRAPPER_TOOLS = ("docker", "tt-smi", "python3")
_WRAPPER_NAME = "run_in_container.sh"
_DEVICE_ROOT = "/dev/tenstorrent"
_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "zsh", "ksh", "fish"})
_COMMAND_PREFIXES = frozenset(
    {"env", "sudo", "command", "timeout", "stdbuf", "nice", "nohup"}
)
_UNKNOWN = object()

# ``None`` denies all guarded operations.  ``hardware`` permits both process
# launch and device opens.  ``wrapper`` carries the canonical pytest base
# directory and permits only a copied wrapper whose launch environment resolves
# every required tool to executable files below that directory.
_permission_scope: contextvars.ContextVar[tuple[str, str | None] | None] = (
    contextvars.ContextVar("hekatus_test_hardware_permission", default=None)
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
        and _scope_name() != "hardware"
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
        fd = int(dir_fd)
        if fd == getattr(os, "AT_FDCWD", -100):
            return os.getcwd()
        return os.readlink(f"/proc/self/fd/{fd}")
    except (OSError, TypeError, ValueError):
        return None


def deny_access() -> contextvars.Token[tuple[str, str | None] | None]:
    """Deny guarded operations for the current test context."""
    return _permission_scope.set(None)


def restore_access(token: contextvars.Token[tuple[str, str | None] | None]) -> None:
    """Restore the permission scope that preceded a fixture."""
    _permission_scope.reset(token)


def permit_hardware() -> contextvars.Token[tuple[str, str | None] | None]:
    """Permit hardware operations after the explicit operator opt-in."""
    return _permission_scope.set(("hardware", None))


def permit_wrapper(
    basetemp: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None = None,
) -> contextvars.Token[tuple[str, str | None] | None]:
    """Permit only safe copied-wrapper launches below the pytest base directory."""
    return _permission_scope.set(("wrapper", _canonical_root(basetemp)))


def _scope_name() -> str | None:
    state = _permission_scope.get()
    return None if state is None else state[0]


def _wrapper_root() -> str | None:
    state = _permission_scope.get()
    return None if state is None else state[1]


def _audit_event(event: str, args: tuple[Any, ...]) -> None:
    scope = _scope_name()
    if event == "open":
        if scope != "hardware" and args and _is_device_path(args[0]):
            raise HardwareAccessViolation("blocked device open")
        return

    if event not in _PROCESS_EVENTS:
        return
    if scope == "hardware":
        return
    if scope == "wrapper" and _contains_wrapper_command(event, args):
        if (
            not _contains_blocked_executable(event, args, _NON_WRAPPER_EXECUTABLES)
            and _wrapper_launch_allowed(event, args, _wrapper_root())
        ):
            return
        raise HardwareAccessViolation("wrapper launch is outside approved stub scope")
    if _contains_blocked_executable(event, args):
        raise HardwareAccessViolation("blocked hardware process")


def _event_payload(
    event: str, args: tuple[Any, ...]
) -> tuple[Any, Any, Any, Any] | None:
    if event == "subprocess.Popen":
        if len(args) < 4:
            return None
        return args[0], args[1], args[2], args[3]
    if event in _POSIX_SPAWN_EVENTS:
        if len(args) < 3:
            return None
        cwd = args[3] if len(args) > 3 else None
        return args[0], args[1], cwd, args[2]
    if event in _SPAWN_EVENTS:
        if len(args) < 3:
            return None
        env = args[3] if len(args) > 3 else None
        return args[1], args[2], None, env
    if event.startswith("os.exec"):
        if len(args) < 2:
            return None
        if event in _EXEC_ENV_EVENTS or (
            event == "os.exec" and len(args) > 2
        ):
            env = args[-1]
        else:
            env = None
        return args[0], args[1], None, env
    if event in {"os.system", "os.popen"}:
        return None, args[0] if args else _UNKNOWN, None, None
    return None


def _contains_wrapper_command(event: str, args: tuple[Any, ...]) -> bool:
    payload = _event_payload(event, args)
    if payload is None:
        return _raw_wrapper_command(event, args)
    executable, argv, _, _ = payload
    return _argv_contains_wrapper(executable) or _argv_contains_wrapper(argv)


def _raw_command_parts(event: str, args: tuple[Any, ...]) -> tuple[Any, Any]:
    if event in _POSIX_SPAWN_EVENTS or event == "subprocess.Popen":
        return args[0] if args else None, args[1] if len(args) > 1 else None
    if event in _SPAWN_EVENTS:
        return args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None
    if event.startswith("os.exec"):
        return args[0] if args else None, args[1] if len(args) > 1 else None
    return None, args[0] if args else None


def _raw_wrapper_command(event: str, args: tuple[Any, ...]) -> bool:
    executable, argv = _raw_command_parts(event, args)
    return _argv_contains_wrapper(executable) or _argv_contains_wrapper(argv)


def _argv_contains_wrapper(value: Any) -> bool:
    if isinstance(value, (str, bytes, os.PathLike)):
        return any(_named_token(token) for token in _shell_tokens(value))
    if isinstance(value, (tuple, list)):
        return any(_argv_contains_wrapper(token) for token in value)
    return False


def _wrapper_launch_allowed(
    event: str, args: tuple[Any, ...], root: str | None
) -> bool:
    if root is None:
        return False
    payload = _event_payload(event, args)
    if payload is None:
        return False
    executable, argv, cwd, env = payload
    wrapper = _direct_wrapper_token(executable, argv)
    path_value = _effective_path(env)
    cwd_value = _effective_cwd(cwd)
    if wrapper is None or path_value is None or cwd_value is None:
        return False
    if not _resolves_executable(wrapper, path_value, cwd_value, root):
        return False
    return all(
        _resolves_executable(tool, path_value, cwd_value, root)
        for tool in _WRAPPER_TOOLS
    )


def _direct_wrapper_token(executable: Any, argv: Any) -> str | None:
    if not _named_token(executable):
        return None
    if not isinstance(argv, (tuple, list)) or not argv:
        return None
    if not _named_token(argv[0]):
        return None
    return _decode_pathlike(executable)


def _named_token(value: Any) -> bool:
    text = _as_text(value)
    return text is not None and _executable_name(text) == _WRAPPER_NAME


def _contains_blocked_executable(
    event: str,
    args: tuple[Any, ...],
    names: frozenset[str] = _BLOCKED_EXECUTABLES,
) -> bool:
    payload = _event_payload(event, args)
    if payload is None:
        executable, argv = _raw_command_parts(event, args)
        return _blocked_token(executable, names) or _blocked_argv(argv, names)
    executable, argv, _, _ = payload
    return _blocked_token(executable, names) or _blocked_argv(argv, names)


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
    text = _as_text(value)
    return text is not None and _executable_name(text) in names


def _as_text(value: Any) -> str | None:
    try:
        value = os.fspath(value)
    except (OSError, TypeError, ValueError, RuntimeError):
        return None
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return value if isinstance(value, str) else None


def _decode_pathlike(value: Any) -> str:
    text = _as_text(value)
    return "" if text is None else text


def _executable_name(value: str) -> str:
    # Audit payloads may contain shell assignments or surrounding whitespace;
    # exact basename matching avoids blocking ordinary names that merely have
    # a guarded token as a substring.
    candidate = value.strip().replace("\\", "/").rstrip("/")
    return posixpath.basename(candidate)


def _effective_path(env: Any) -> str | None:
    if env is _UNKNOWN:
        return None
    source = os.environ if env is None else env
    try:
        items = source.items()
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError):
        return None
    path_value: str | None = None
    path_seen = False
    try:
        for key, value in items:
            key_text = _as_text(key)
            value_text = _as_text(value)
            if key_text is None or value_text is None:
                return None
            if key_text == "PATH":
                if path_seen:
                    return None
                path_seen = True
                path_value = value_text
    except (OSError, TypeError, ValueError, RuntimeError):
        return None
    return os.defpath if path_value is None else path_value


def _effective_cwd(cwd: Any) -> str | None:
    if cwd is _UNKNOWN:
        return None
    try:
        current = os.getcwd()
        if cwd is None:
            return os.path.realpath(current)
        cwd_text = _as_text(cwd)
        if not cwd_text:
            return None
        if not os.path.isabs(cwd_text):
            cwd_text = os.path.join(current, cwd_text)
        return os.path.realpath(cwd_text)
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _canonical_root(value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        root = os.path.realpath(os.path.abspath(text))
        return root if os.path.isdir(root) else None
    except (OSError, TypeError, ValueError, RuntimeError):
        return None


def _resolves_executable(
    program: Any, path_value: str, cwd: str, root: str
) -> bool:
    text = _as_text(program)
    if not text or "\x00" in text:
        return False
    if os.path.isabs(text):
        candidates = (text,)
    elif "/" in text.replace("\\", "/"):
        candidates = (os.path.join(cwd, text),)
    else:
        candidates = tuple(
            os.path.join(cwd if not entry else entry, text)
            if os.path.isabs(entry) or not entry
            else os.path.join(cwd, entry, text)
            for entry in path_value.split(os.pathsep)
        )
    for candidate in candidates:
        result = _check_executable_candidate(candidate, root)
        if result is True:
            return True
        if result is False:
            return False
    return False


def _check_executable_candidate(candidate: str, root: str) -> bool | None:
    try:
        if not os.path.lexists(candidate):
            return None
        metadata = os.stat(candidate)
        if not stat.S_ISREG(metadata.st_mode) or not os.access(candidate, os.X_OK):
            return None
        canonical = os.path.realpath(candidate)
        if not _path_within(canonical, root):
            return False
    except (OSError, TypeError, ValueError):
        return None
    return True


def _path_within(candidate: str, root: str) -> bool:
    try:
        return os.path.commonpath((os.path.abspath(candidate), root)) == root
    except (OSError, ValueError):
        return False


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
