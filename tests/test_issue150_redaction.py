"""Board-free Issue #150 invariants, enumerated before implementation.

1. Only slash tokens whose posixpath.normpath is at/below /work, /out, or
   /dev/tenstorrent, without any '..' component, may remain. Whitespace and
   quotes delimit tokens; commas, semicolons, and parentheses never do.
2. Only public HTTP(S) URLs retain payloads; file:// and scheme:/// do not.
3. Hostname, FQDN, and first label are case-insensitive whole words, even at
   length one, including inside allowed paths and URLs; larger words remain.
4. One ingress handles str/bytes and every diagnostic sink, before tail bounds:
   stdout/stderr files, tails, failure fields, record reason/trace_error,
   pre/post snapshot diagnostics, stored captures, and runner-result diagnostics.
   Non-diagnostic structured evidence remains unchanged. Writers are memory sinks.
5. Relative paths, exception classes, ModuleNotFoundError text, and image
   digests remain readable unless they contain a runtime host word.
6. Empty, binary, mixed-encoding, surrogate, and unprintable-object inputs never
   raise; output is UTF-8 encodable. Reapplying redaction is stable and cannot
   restore user-supplied placeholder text as a protected path or URL.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import issue146


@pytest.fixture(autouse=True)
def diagnostic_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(issue146.socket, "gethostname", lambda: "q")
    monkeypatch.setattr(issue146.socket, "getfqdn", lambda: "q.lab.example")


def redact(value: object, *, host_tokens: tuple[str, ...] | None = None) -> str:
    return issue146._redact_diagnostic_text(
        value, output_dir=Path("/"), host_tokens=host_tokens
    )


@pytest.mark.parametrize(
    "token",
    [
        "/private/secret", "//private/secret", "///private/secret", "////private/secret",
        "/work/../private/secret", "/out/../../private/secret",
        "/dev/tenstorrent/../../../private/secret", "/work/sub/../safe.py",
        "/outbox/secret", "/worktree/secret", "/dev/tenstorrent-extra/secret",
        "/日本語/secret", "/éclair/secret", "/@private/secret", "/!private/secret",
        "/private/secret,comma", "/private/secret;semicolon", "/private/(secret)",
        "/private/secret),/other/secret", "/", "//work/safe.py",
    ],
)
@pytest.mark.parametrize("quote", ["", "'", '"'])
def test_absolute_path_allowlist_rejects_entire_token(token: str, quote: str) -> None:
    assert redact(f"before {quote}{token}{quote} after") == (
        f"before {quote}<redacted-path>{quote} after"
    )


@pytest.mark.parametrize(
    "token",
    [
        "/work", "/out", "/dev/tenstorrent", "/work/tools/issue146.py",
        "/out/result.json", "/dev/tenstorrent/1", "/work/./source.py",
        "/out//result.json", "///work/tools/source.py", "/work/日本語/(source),v;1.py",
        "relative/tools/issue146.py", "relative/tmp/source.py", "../tools/source.py",
        "ModuleNotFoundError: No module named 'scipy'", "TimeoutExpired OSError ValueError",
        "sha256:" + "a" * 64,
        "ghcr.io/public/image@sha256:" + "a" * 64,
    ],
)
def test_allowlisted_context_remains(token: str) -> None:
    assert redact(token) == token


@pytest.mark.parametrize("quote", ["'", '"'])
def test_whitespace_and_quotes_are_the_only_path_token_delimiters(quote: str) -> None:
    text = f"File {quote}/private/secret space,/second;secret(third){quote}"
    assert redact(text) == f"File {quote}<redacted-path> space,<redacted-path>{quote}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("file:///private/secret", "file://<redacted-path>"),
        ("FILE://q/private/secret", "FILE://<redacted-path>"),
        ("sqlite:////private/(secret),v;1", "sqlite://<redacted-path>"),
        ("custom:///work/safe.py", "custom://<redacted-path>"),
        ("https:///private/secret", "https://<redacted-path>"),
        ("custom://q/private/secret", "custom://<redacted-path>"),
        ("file://", "file://<redacted-path>"),
        ("https://example.test/private/(source),v;1", "https://example.test/private/(source),v;1"),
        ("HTTP://example.test/source", "HTTP://example.test/source"),
        ("https://q.lab.example/source", "https://<redacted>/source"),
        ("https://example.test/q/source", "https://example.test/<redacted>/source"),
    ],
)
def test_url_allowlist(text: str, expected: str) -> None:
    assert redact(text) == expected


@pytest.mark.parametrize("hostname", ["q", "q.lab.example", "Q.LAB.EXAMPLE"])
def test_hostname_words_of_any_length(hostname: str) -> None:
    assert redact(f"host={hostname} ({hostname})") == "host=<redacted> (<redacted>)"


def test_hostname_first_labels_and_unicode_word_boundaries() -> None:
    assert issue146._diagnostic_host_tokens() == ("q.lab.example", "q")
    assert redact("q q.lab.example q.other q-name aq qa q_name éq qé") == (
        "<redacted> <redacted> <redacted>.other <redacted>-name aq qa q_name éq qé"
    )
    assert redact("/work/q/source.py relative/q.py") == (
        "/work/<redacted>/source.py relative/<redacted>.py"
    )


class Unprintable:
    def __str__(self) -> str:
        raise ValueError("unprintable diagnostic")


@pytest.mark.parametrize(
    "value",
    [None, "", b"", bytes(range(256)), b"\xff/private/secret \xc3\xa9\x80",
     "\ud800 /private/secret", "\x00/private/secret\x00", Unprintable(), 12],
)
def test_ingress_is_total_and_utf8_encodable(value: object) -> None:
    output = redact(value)
    assert isinstance(output, str)
    output.encode("utf-8")
    assert "/private/secret" not in output


@pytest.mark.parametrize("host", ["q", "redacted", "path"])
def test_redaction_is_stable_without_placeholder_restoration(host: str) -> None:
    text = (
        "\\x00issue146-url-0\\x00 \\x00issue146-safe-path-0\\x00 "
        f"/private/secret /work/{host}/safe https://example.test/{host} "
        "file:///private/secret"
    )
    once = redact(text, host_tokens=(host,))
    assert redact(once, host_tokens=(host,)) == once
    assert "/private/secret" not in once
    assert "\\x00issue146-url-0\\x00" in once
    # User-supplied placeholder-like text is ordinary text, so host words in
    # it still disappear; unlike the old substitution scheme it is not restored.
    expected_placeholder = (
        "\\x00issue146-safe-<redacted>-0\\x00"
        if host == "path" else "\\x00issue146-safe-path-0\\x00"
    )
    assert expected_placeholder in once


@pytest.mark.parametrize("as_bytes", [False, True])
def test_all_persisted_diagnostic_sinks_use_ingress(
    monkeypatch: pytest.MonkeyPatch, as_bytes: bool,
) -> None:
    """Observe the actual persistence seams, not a JSON-wide string scrub."""
    text = (
        "//private/secret /work/../private/secret /日本語/secret "
        "'/private/(secret),v;1' q.lab.example q "
        "/work/tools/issue146.py relative/tools/issue146.py "
        "/work/tmp/source.py https://example.test/home/remote "
        "ModuleNotFoundError: No module named 'scipy' sha256:" + "a" * 64
    )
    expected = (
        "<redacted-path> <redacted-path> <redacted-path> "
        "'<redacted-path>' <redacted> <redacted> "
        "/work/tools/issue146.py relative/tools/issue146.py "
        "/work/tmp/source.py https://example.test/home/remote "
        "ModuleNotFoundError: No module named 'scipy' sha256:" + "a" * 64
    )
    diagnostic = text.encode() if as_bytes else text
    persisted: dict[str, object] = {}

    def byte_sink(path: Path, payload: bytes, **_kwargs: object) -> bytes:
        persisted[path.name] = payload
        return payload

    def json_sink(path: Path, payload: object) -> bytes:
        # Keep the production metadata/JSON transform; replace only file I/O.
        safe = issue146._safe_json_value(payload)
        persisted[path.name] = copy.deepcopy(safe)
        return json.dumps(safe).encode()

    monkeypatch.setattr(issue146, "_write_bytes_artifact", byte_sink)
    monkeypatch.setattr(issue146, "_write_artifact", json_sink)
    monkeypatch.setattr(
        issue146, "_safe_artifact_path", lambda output_dir, name, **_kwargs: output_dir / name
    )
    prepared = SimpleNamespace(
        output_dir=Path("/out"), wrapper_stdout_path=Path("/out/stdout.txt"),
        wrapper_stderr_path=Path("/out/stderr.txt"),
    )
    state = issue146._ExecutionState(
        captures=[], publication_state=None, trace_stop=threading.Event(),
        stdout=diagnostic, stderr=diagnostic,
    )
    wrapper = issue146._persist_wrapper_diagnostics(prepared, state)
    assert persisted["stdout.txt"] == persisted["stderr.txt"] == expected.encode()
    assert wrapper["stdout_tail"] == wrapper["stderr_tail"] == [expected]
    assert wrapper["stdout_byte_count"] == wrapper["stderr_byte_count"] == len(expected.encode())
    failure = {
        "message": diagnostic, "wrapper_error": diagnostic, "diagnostics_error": diagnostic,
        "exception_type": "ModuleNotFoundError", "returncode": 17,
        "wrapper_diagnostics": {"stdout_tail": [diagnostic], "stderr_tail": [diagnostic]},
    }
    record = {
        "reason": diagnostic, "trace_error": diagnostic, "failure": failure,
        "wrapper_diagnostics": failure["wrapper_diagnostics"],
        "pre_run_snapshot": {"configuration": {"diagnostic": diagnostic}},
        "post_run_snapshot": {"reason": diagnostic},
        "runner_result": {
            "failure": failure, "wrapper_diagnostics": failure["wrapper_diagnostics"],
            "reason": diagnostic, "trace_error": diagnostic,
        },
        "environment": {"mode": "q", "device_node": "/dev/tenstorrent/1"},
    }
    issue146._persist_record(Path("/out"), "test150", record)
    saved = persisted["run-test150.json"]
    assert saved == record
    assert saved["reason"] == saved["trace_error"] == expected
    assert saved["pre_run_snapshot"]["configuration"]["diagnostic"] == expected
    assert saved["post_run_snapshot"]["reason"] == expected
    for container in (saved, saved["runner_result"]):
        assert container["reason"] == container["trace_error"] == expected
        for field in ("message", "wrapper_error", "diagnostics_error"):
            assert container["failure"][field] == expected
        assert container["failure"]["exception_type"] == "ModuleNotFoundError"
        assert container["failure"]["returncode"] == 17
        for diagnostics in (container, container["failure"]):
            for field in ("stdout_tail", "stderr_tail"):
                assert diagnostics["wrapper_diagnostics"][field] == [expected]
    assert saved["environment"] == {"mode": "q", "device_node": "/dev/tenstorrent/1"}
    for stage in ("pre_run", "post_run"):
        capture = {
            "reason": diagnostic, "configuration": {"diagnostic": diagnostic},
            "snapshot": {"device_node": "/dev/tenstorrent/1"},
        }
        descriptor = issue146._store_capture(
            Path("/out"), run_id="test150", stage=stage, capture=capture
        )
        payload = persisted[descriptor["path"]]
        assert payload["reason"] == payload["configuration"]["diagnostic"] == expected
        assert payload["snapshot"] == capture["snapshot"]
        raw = json.dumps(payload).encode()
        assert descriptor["sha256"] == hashlib.sha256(raw).hexdigest()
        assert descriptor["byte_count"] == len(raw)


def test_runner_ingestion_sanitizes_before_structural_metadata_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = (
        "ModuleNotFoundError: No module named 'scipy'; /home/private/secret q "
        "relative/tmp/source.py https://example.test/home/remote"
    )
    result = {"failure": {"message": diagnostic}, "record_schema": "test"}
    monkeypatch.setattr(
        issue146, "_resolve_result", lambda *_args, **_kwargs: (Path("result.json"), "result.json")
    )
    monkeypatch.setattr(issue146, "_read_json", lambda *_args, **_kwargs: (result, b"{}"))

    def reject_result(*_args: object, **_kwargs: object) -> None:
        raise issue146.Issue146Error("injected validation failure")

    monkeypatch.setattr(issue146, "_validate_runner_result", reject_result)
    prepared = SimpleNamespace(
        output_dir=Path("/out"), record={"provenance": {}, "run": {"mode": "smoke"}},
        safe_run_id="test150",
    )
    state = issue146._ExecutionState(
        captures=[], publication_state=None, trace_stop=threading.Event(), completed=None,
    )
    with pytest.raises(issue146.Issue146Error, match="injected validation failure"):
        issue146._ingest_result(prepared, state, before=set(), plan={}, configuration=None)
    assert prepared.record["runner_result"]["failure"]["message"] == (
        "ModuleNotFoundError: No module named 'scipy'; <redacted-path> <redacted> "
        "relative/tmp/source.py https://example.test/home/remote"
    )


@pytest.mark.parametrize("as_bytes", [False, True])
def test_redacts_full_text_before_tail_bounds(
    monkeypatch: pytest.MonkeyPatch, as_bytes: bool,
) -> None:
    """A path beyond the old truncation boundary cannot expose a partial token."""
    text = "discarded //private/secret\n" + "ordinary line\n" * 25
    text += "x=" + "/private/" + "s" * 2100 + ",secret"
    memory: dict[str, bytes] = {}

    def sink(path: Path, payload: bytes, **_kwargs: object) -> bytes:
        memory[path.name] = payload
        return payload

    monkeypatch.setattr(issue146, "_write_bytes_artifact", sink)
    prepared = SimpleNamespace(
        output_dir=Path("/out"), wrapper_stdout_path=Path("/out/stdout.txt"),
        wrapper_stderr_path=Path("/out/stderr.txt"),
    )
    state = issue146._ExecutionState(
        captures=[], publication_state=None, trace_stop=threading.Event(),
        stdout=text.encode() if as_bytes else text, stderr=b"",
    )
    diagnostics = issue146._persist_wrapper_diagnostics(prepared, state)
    assert memory["stdout.txt"].startswith(b"discarded <redacted-path>\n")
    assert diagnostics["stdout_tail"] == ["ordinary line"] * 19 + ["x=<redacted-path>"]
    assert diagnostics["stderr_tail"] == []
