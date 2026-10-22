r"""Board-free Issue #150 invariants, enumerated before implementation.

1. Tokens are maximal sequences excluding whitespace and either quote. Classify
   whole scheme://payload URL tokens first. A relative token fully matches
   ^(?:\\.\\.?/)?(?:[\\w.+~-]+/)*[\\w.+~-]*$ and starts with neither '-' nor '~'.
   Preserve relative tokens; otherwise split at the first slash or a tilde at
   token start/after a non-word character. Keep the lexical prefix, not a list
   of introducers. Suffix S remains only if it fully matches ^[\\w./+~-]+$,
   has no '..' component, and posixpath.normpath(S) is at/below /work, /out, or
   /dev/tenstorrent. Otherwise replace all of S with <redacted-path>.
   Property tests use host_tokens=(): every slash-bearing output token is an
   unchanged relative token, a public HTTP(S) URL, or prefix plus an allowed
   pure suffix/redaction marker. Every input relative token stays unchanged.
2. Only public HTTP(S) URLs retain payloads; file:// and scheme:/// do not.
   Public authority must be nonempty and contain no '@'; authority ends at a
   slash, query, or fragment. Query and fragment contents have no extra filter.
3. After token classification, mask hostname, FQDN, and first label across the
   entire text as case-insensitive whole words even at length one. Larger words
   and output markers remain. Host masking is tested separately from grammar.
4. One ingress handles str/bytes and every diagnostic sink, before tail bounds:
   stdout/stderr files, tails, failure fields, record reason/trace_error,
   pre/post snapshot diagnostics, stored captures, and runner-result diagnostics.
   Nested diagnostic mappings/lists in snapshots, records, and runner results
   share one host-token tuple per persistence unit. With host_tokens=None,
   _safe_json_value obtains it once at the outermost call and propagates it.
   Non-diagnostic structured evidence remains unchanged. Writers are memory sinks.
5. Admitted relative paths, exception classes, ModuleNotFoundError text, and
   standalone image digests remain readable unless they contain a runtime host word.
6. Empty, binary, mixed-encoding, surrogate, and unprintable-object inputs never
   raise; output is UTF-8 encodable. Markers remain stable; no user-supplied
   placeholder can restore a protected path or URL. A host-masked path with
   angle-bracket markers may be conservatively removed by a later pure-path pass.
   Nested diagnostic fields preserve None/bool/int/float JSON scalars; other
   unsupported objects become <unsupported-diagnostic> without invoking __str__.
   Secret-returning and throwing objects must be safe at record/snapshot/runner
   result persistence boundaries. The unsupported marker is stable output syntax.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import random
import re
import string
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


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (".", "."), ("..", ".."), ("a/b", "a/b"), ("./a", "./a"),
        ("../a/b.py", "../a/b.py"), ("a~b/c", "a~b/c"),
        ("tools/issue146.py", "tools/issue146.py"),
        ("-I/home/example-user/x", "-I<redacted-path>"),
        ("CFLAGS=-I/home/example-user/x", "CFLAGS=-I<redacted-path>"),
        ("PATH=/home/example-user/bin:/usr/bin", "PATH=<redacted-path>"),
        ("-o/out/x.json:/home/example-user", "-o<redacted-path>"),
        ("--mount=/work/x", "--mount=/work/x"),
        ("(/home/example-user/f.txt),", "(<redacted-path>"),
        ("foo/bar:/home/example-user", "foo<redacted-path>"),
        ("key:/out/x", "key:/out/x"), ("中文=/work/日本語+x", "中文=/work/日本語+x"),
        ("key=/work/x,other", "key=<redacted-path>"),
        ("key=(/work/x)", "key=(<redacted-path>"),
        ("CFLAGS=-I/out/x.json", "CFLAGS=-I/out/x.json"),
        ("x=~example/x", "x=<redacted-path>"),
        ("-I~/x", "-I~<redacted-path>"), ("word~example/x", "word~example/x"),
        ("/out/data;stuff", "<redacted-path>"),
        ("https://example.test/x", "https://example.test/x"),
        ("file:///private", "file://<redacted-path>"),
        ("url=https://example.test/x", "url=https:<redacted-path>"),
        ("https://user:synthetic@example.test/x", "https://<redacted-path>"),
    ],
)
def test_whole_token_grammar_examples(
    text: str, expected: str, quote: str, as_bytes: bool,
) -> None:
    wrapped = f"before {quote}{text}{quote} after"
    assert redact(wrapped.encode() if as_bytes else wrapped, host_tokens=()) == (
        f"before {quote}{expected}{quote} after"
    )


# Independent specification predicates: do not call the production classifier.
_PROPERTY_TOKEN_RE = re.compile(r"[^\s\"']+")
_PROPERTY_RELATIVE_RE = re.compile(r"(?:\.\.?/)?(?:[\w.+~-]+/)*[\w.+~-]*")
_PROPERTY_PURE_RE = re.compile(r"[\w./+~-]+")
_PROPERTY_URL_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://(.*)")


def _property_relative(token: str) -> bool:
    return bool(_PROPERTY_RELATIVE_RE.fullmatch(token)) and not token.startswith(("-", "~"))


def _property_allowed_suffix(suffix: str) -> bool:
    normalized = posixpath.normpath(suffix)
    return (
        bool(_PROPERTY_PURE_RE.fullmatch(suffix))
        and ".." not in suffix.split("/")
        and any(
            normalized == root or normalized.startswith(root + "/")
            for root in ("/work", "/out", "/dev/tenstorrent")
        )
    )


def _assert_property_output_token(original: str, output: str) -> None:
    if "/" not in output:
        return
    if _property_relative(output):
        assert output == original
        return
    url = _PROPERTY_URL_RE.fullmatch(output)
    if url:
        scheme, payload = url.groups()
        authority = re.split(r"[/?#]", payload, maxsplit=1)[0]
        # A redacted URL keeps scheme:// as its syntactic prefix (case c).
        if payload == "<redacted-path>":
            return
        assert scheme.lower() in {"http", "https"} and authority and "@" not in authority
        assert output == original
        return
    assert _property_allowed_suffix(output[output.index("/"):])


def test_seeded_whole_token_output_grammar_property() -> None:
    rng = random.Random(150)
    alphabet = string.ascii_letters + string.digits + "._+~-/=:,;()漢字é \t\n\"'"
    path_alphabet = "abcXYZ019._+~-/=:,;()漢字é"
    urls = (
        "https://example.test/path", "http://example.test?mail=a@b",
        "file:///private", "custom:///work/x", "https:///private",
        "https://user:synthetic@example.test/x",
    )
    tokens_checked = relative_checked = 0
    for _ in range(6000):
        raw = "".join(rng.choices(alphabet, k=rng.randrange(0, 180)))
        relative = rng.choice((".", "..", "dir", "日本語", "a~b")) + "/" + rng.choice(
            ("x.py", "é", "child/data", "a+b", "a~b")
        )
        prefix = "".join(rng.choices("ABCxyz_=:,-;(漢~", k=rng.randrange(15)))
        suffix = rng.choice(("/work/", "/out/", "/dev/tenstorrent/", "/private/"))
        suffix += "".join(rng.choices(path_alphabet, k=rng.randrange(1, 40)))
        separator = rng.choice((" ", "'\t", '\"\n'))
        text = separator.join((raw, relative, prefix + suffix, rng.choice(urls)))
        output = redact(text, host_tokens=())
        originals = _PROPERTY_TOKEN_RE.findall(text)
        outputs = _PROPERTY_TOKEN_RE.findall(output)
        assert _PROPERTY_TOKEN_RE.split(output) == _PROPERTY_TOKEN_RE.split(text)
        for original, transformed in zip(originals, outputs, strict=True):
            tokens_checked += 1
            if _property_relative(original):
                relative_checked += 1
                assert transformed == original
            _assert_property_output_token(original, transformed)
    assert relative_checked >= 6000
    print(f"grammar_property seed=150 texts=6000 tokens={tokens_checked} relative={relative_checked}")


def test_host_masking_follows_classification_and_preserves_markers() -> None:
    text = "CFLAGS=-I/private/x /work/q/x relative/q.py https://q.lab.example/x"
    classified = "CFLAGS=-I<redacted-path> /work/q/x relative/q.py https://q.lab.example/x"
    assert redact(text, host_tokens=()) == classified
    assert redact(text) == (
        "CFLAGS=-I<redacted-path> /work/<redacted>/x relative/<redacted>.py https://<redacted>/x"
    )
    markers = "<redacted-path> <unsupported-diagnostic>"
    assert redact(markers, host_tokens=("redacted", "path", "unsupported", "diagnostic")) == markers


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
    # The approved pure-token grammar rejects punctuation in path suffixes and
    # non-URL registry references; the standalone digest case still remains.
    grammar_expectations = {
        "/work/日本語/(source),v;1.py": "<redacted-path>",
        "ghcr.io/public/image@sha256:" + "a" * 64: "ghcr.io<redacted-path>",
    }
    assert redact(token) == grammar_expectations.get(token, token)


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


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize(
    "token", ["~/.local/bin/uv", "~alice/x", "~日本語/(source),v;1", "~/x;comma,(tail)"],
)
def test_tilde_leading_tokens_are_user_specific(
    token: str, quote: str, as_bytes: bool,
) -> None:
    text = f"before {quote}{token}{quote} after"
    value = text.encode() if as_bytes else text
    assert redact(value) == f"before {quote}<redacted-path>{quote} after"


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("token", ["a~b/c", "relative/a~b/c", "word~alice/x", "/work/a~b/c"])
def test_word_internal_tildes_remain(token: str, as_bytes: bool) -> None:
    assert redact(token.encode() if as_bytes else token) == token


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("http://user:example-pass@example.test/source", "http://<redacted-path>"),
        ("https://user@example.test/source?x=1#part", "https://<redacted-path>"),
        ("HTTPS://user:example-pass@q.lab.example/source", "HTTPS://<redacted-path>"),
        ("https://@example.test", "https://<redacted-path>"),
        ("http://example.test/source", "http://example.test/source"),
        ("https://example.test?mail=a@b", "https://example.test?mail=a@b"),
        ("https://example.test#mail=a@b", "https://example.test#mail=a@b"),
        ("https://example.test/source?mail=a@b#c@d", "https://example.test/source?mail=a@b#c@d"),
        ("https://q.lab.example?mail=a@b", "https://<redacted>?mail=a@b"),
    ],
)
def test_public_url_authority_excludes_userinfo_without_query_filtering(
    text: str, expected: str, as_bytes: bool,
) -> None:
    assert redact(text.encode() if as_bytes else text) == expected


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("sink", ["pre_run", "post_run", "record", "runner_result"])
def test_nested_diagnostics_mask_hosts_once_per_persistence(
    monkeypatch: pytest.MonkeyPatch, as_bytes: bool, sink: str,
) -> None:
    calls = {"hostname": 0, "fqdn": 0}

    def hostname() -> str:
        calls["hostname"] += 1
        return "q"

    def fqdn() -> str:
        calls["fqdn"] += 1
        return "q.lab.example"

    monkeypatch.setattr(issue146.socket, "gethostname", hostname)
    monkeypatch.setattr(issue146.socket, "getfqdn", fqdn)
    monkeypatch.setattr(
        issue146, "_safe_artifact_path", lambda output_dir, name, **_kwargs: output_dir / name
    )
    memory: dict[str, object] = {}

    def json_sink(
        path: Path, payload: object, *, host_tokens: tuple[str, ...] | None = None,
    ) -> bytes:
        safe = (
            issue146._safe_json_value(payload)
            if host_tokens is None
            else issue146._safe_json_value(payload, host_tokens=host_tokens)
        )
        memory[path.name] = copy.deepcopy(safe)
        return json.dumps(safe).encode()

    monkeypatch.setattr(issue146, "_write_artifact", json_sink)
    text = "q q.lab.example /private/secret relative/tools/source.py"
    diagnostic = text.encode() if as_bytes else text
    metadata = {"mode": "q", "name": "q.lab.example", "device_node": "/dev/tenstorrent/1"}
    nested = {
        "reason": diagnostic,
        "rows": [{"diagnostic": diagnostic}, {"failure": {"message": diagnostic}}],
        "metadata": metadata,
    }
    if sink in {"pre_run", "post_run"}:
        descriptor = issue146._store_capture(
            Path("/out"), run_id="nested150", stage=sink, capture={"snapshot": nested}
        )
        payload = memory[descriptor["path"]]
        saved = payload["snapshot"]
        raw = json.dumps(payload).encode()
        assert descriptor["sha256"] == hashlib.sha256(raw).hexdigest()
        assert descriptor["byte_count"] == len(raw)
    else:
        record = {"runner_result": {"snapshot": nested}} if sink == "runner_result" else nested
        issue146._persist_record(Path("/out"), "nested150", record)
        payload = memory["run-nested150.json"]
        assert record == payload
        saved = payload["runner_result"]["snapshot"] if sink == "runner_result" else payload
    expected = "<redacted> <redacted> <redacted-path> relative/tools/source.py"
    assert saved["reason"] == expected
    assert saved["rows"][0]["diagnostic"] == expected
    assert saved["rows"][1]["failure"]["message"] == expected
    assert saved["metadata"] == metadata
    assert calls == {"hostname": 1, "fqdn": 1}


@pytest.mark.parametrize("as_bytes", [False, True])
def test_recursive_json_default_host_tokens_are_acquired_once(
    monkeypatch: pytest.MonkeyPatch, as_bytes: bool,
) -> None:
    calls: list[str] = []

    def tokens() -> tuple[str, ...]:
        calls.append("lookup")
        return ("q",)

    monkeypatch.setattr(issue146, "_diagnostic_host_tokens", tokens)
    text = b"q /private/secret" if as_bytes else "q /private/secret"
    value = {"reason": text, "rows": [{"diagnostic": text}], "mode": "q"}
    assert issue146._safe_json_value(value) == {
        "reason": "<redacted> <redacted-path>",
        "rows": [{"diagnostic": "<redacted> <redacted-path>"}], "mode": "q",
    }
    assert calls == ["lookup"]


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("-I/home/example-user/include", "-I<redacted-path>"),
        ("--include=/home/example-user", "--include=<redacted-path>"),
        ("-L/日本語/(source),v;1", "-L<redacted-path>"),
        ("---cache~example-user/cache", "---cache~example-user<redacted-path>"),
        ("-I//private/secret", "-I<redacted-path>"),
        ("-I/work/../private/secret", "-I<redacted-path>"),
        ("-I~/.local/bin/uv", "-I~<redacted-path>"),
        ("-o/out/x.json", "-o/out/x.json"),
        ("-I/work/source", "-I/work/source"),
        ("---node/dev/tenstorrent/1", "---node/dev/tenstorrent/1"),
        ("--include=/out/x", "--include=/out/x"),
        ("--mount /work/x", "--mount /work/x"),
    ],
)
def test_option_attached_paths_use_allowlist(
    text: str, expected: str, quote: str, as_bytes: bool,
) -> None:
    # The first case reproduces the CodeRabbit include-option counterexample
    # with a synthetic user path, not a path obtained from the execution host.
    wrapped = f"before {quote}{text}{quote} after"
    assert redact(wrapped.encode() if as_bytes else wrapped) == (
        f"before {quote}{expected}{quote} after"
    )


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize(
    "text", ["tools/issue146.py", "enodia/tt/x", "./rel", "../rel", "a~b/c", "a-I/private"],
)
def test_relative_tokens_remain_distinct_from_option_paths(text: str, as_bytes: bool) -> None:
    assert redact(text.encode() if as_bytes else text) == text


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("introducer", ["=", ":", ",", "("])
def test_punctuation_introduced_absolute_paths_are_still_detected(
    introducer: str, as_bytes: bool,
) -> None:
    text = f"argument{introducer}/private/secret"
    assert redact(text.encode() if as_bytes else text) == f"argument{introducer}<redacted-path>"


class DiagnosticObject:
    def __init__(self, *, throwing: bool) -> None:
        self.throwing = throwing
        self.string_calls = 0

    def __str__(self) -> str:
        self.string_calls += 1
        if self.throwing:
            raise ValueError("diagnostic object cannot be converted")
        return "/private/secret q"


@pytest.fixture
def memory_json_artifacts(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    memory: dict[str, object] = {}

    def writer(
        path: Path, payload: object, *, host_tokens: tuple[str, ...] | None = None,
    ) -> bytes:
        safe = issue146._safe_json_value(payload, host_tokens=host_tokens)
        raw = json.dumps(safe).encode()
        memory[path.name] = json.loads(raw)
        return raw

    monkeypatch.setattr(issue146, "_write_artifact", writer)
    monkeypatch.setattr(
        issue146, "_safe_artifact_path", lambda output_dir, name, **_kwargs: output_dir / name
    )
    return memory


@pytest.mark.parametrize("sink", ["pre_run", "post_run", "record", "runner_result"])
@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CFLAGS=-I/home/example-user/include", "CFLAGS=-I<redacted-path>"),
        ("CFLAGS=-I/out/include", "CFLAGS=-I/out/include"),
        ("CFLAGS=-I/out/include:/private/secret", "CFLAGS=-I<redacted-path>"),
        ("PATH=/home/example-user/bin:/usr/bin", "PATH=<redacted-path>"),
        ("FLAGS=(/private/source),", "FLAGS=(<redacted-path>"),
    ],
)
def test_assignment_option_failure_message_at_persistence(
    memory_json_artifacts: dict[str, object], sink: str, as_bytes: bool,
    text: str, expected: str,
) -> None:
    value = text.encode() if as_bytes else text
    nested = {"rows": [{"failure": {"message": value}}], "metadata": {"mode": "q"}}
    if sink in {"pre_run", "post_run"}:
        descriptor = issue146._store_capture(
            Path("/out"), run_id="grammar150", stage=sink, capture={"snapshot": nested}
        )
        saved = memory_json_artifacts[descriptor["path"]]["snapshot"]
    else:
        record = {"runner_result": nested} if sink == "runner_result" else nested
        issue146._persist_record(Path("/out"), "grammar150", record)
        payload = memory_json_artifacts["run-grammar150.json"]
        assert payload == record
        saved = payload["runner_result"] if sink == "runner_result" else payload
    assert saved["rows"][0]["failure"]["message"] == expected
    assert saved["metadata"] == {"mode": "q"}


@pytest.mark.parametrize("sink", ["pre_run", "post_run", "record", "runner_result"])
@pytest.mark.parametrize(
    ("kind", "scalar"),
    [("secret", None), ("throwing", None), ("json", None), ("json", True),
     ("json", False), ("json", 0), ("json", -17), ("json", 2.5)],
)
def test_nested_diagnostic_objects_and_json_scalars_at_persistence(
    memory_json_artifacts: dict[str, object], sink: str, kind: str, scalar: object,
) -> None:
    diagnostic_object = DiagnosticObject(throwing=kind == "throwing") if kind != "json" else None
    value = diagnostic_object if diagnostic_object is not None else scalar
    expected = "<unsupported-diagnostic>" if diagnostic_object is not None else scalar
    nested = {
        "reason": value,
        "rows": [{"diagnostic": value}, {"failure": {"message": value}}],
        "metadata": {"mode": "q", "device_node": "/dev/tenstorrent/1"},
    }
    if sink in {"pre_run", "post_run"}:
        descriptor = issue146._store_capture(
            Path("/out"), run_id="objects150", stage=sink, capture={"snapshot": nested}
        )
        saved = memory_json_artifacts[descriptor["path"]]["snapshot"]
    else:
        record = {"runner_result": {"snapshot": nested}} if sink == "runner_result" else nested
        issue146._persist_record(Path("/out"), "objects150", record)
        payload = memory_json_artifacts["run-objects150.json"]
        assert record == payload
        saved = payload["runner_result"]["snapshot"] if sink == "runner_result" else payload
    for actual in (
        saved["reason"], saved["rows"][0]["diagnostic"], saved["rows"][1]["failure"]["message"],
    ):
        assert actual == expected
        assert type(actual) is type(expected)
    assert saved["metadata"] == {"mode": "q", "device_node": "/dev/tenstorrent/1"}
    if diagnostic_object is not None:
        assert diagnostic_object.string_calls == 0


@pytest.mark.parametrize("host", ["q", "unsupported", "diagnostic"])
def test_unsupported_diagnostic_marker_survives_host_masking(host: str) -> None:
    value = DiagnosticObject(throwing=True)
    once = redact(value, host_tokens=(host,))
    assert once == "<unsupported-diagnostic>"
    assert redact(once, host_tokens=(host,)) == once
    assert value.string_calls == 0


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
    # Marker brackets are not pure path characters. A later lexical pass may
    # conservatively remove this suffix, but must never restore private text.
    twice = redact(once, host_tokens=(host,))
    assert twice == once.replace("/work/<redacted>/safe", "<redacted-path>")
    assert redact(twice, host_tokens=(host,)) == twice
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
        "~/.local/bin/uv ~alice/x http://example-user:example-pass@example.test/source "
        "-I/home/example-user/include -o/out/x.json "
        "/work/tools/issue146.py relative/tools/issue146.py "
        "/work/tmp/source.py https://example.test/home/remote "
        "ModuleNotFoundError: No module named 'scipy' sha256:" + "a" * 64
    )
    expected = (
        "<redacted-path> <redacted-path> <redacted-path> "
        "'<redacted-path>' <redacted> <redacted> "
        "<redacted-path> <redacted-path> http://<redacted-path> "
        "-I<redacted-path> -o/out/x.json "
        "/work/tools/issue146.py relative/tools/issue146.py "
        "/work/tmp/source.py https://example.test/home/remote "
        "ModuleNotFoundError: No module named 'scipy' sha256:" + "a" * 64
    )
    diagnostic = text.encode() if as_bytes else text
    persisted: dict[str, object] = {}

    def byte_sink(path: Path, payload: bytes, **_kwargs: object) -> bytes:
        persisted[path.name] = payload
        return payload

    def json_sink(
        path: Path, payload: object, *, host_tokens: tuple[str, ...] | None = None,
    ) -> bytes:
        # Keep the production metadata/JSON transform; replace only file I/O.
        safe = (
            issue146._safe_json_value(payload)
            if host_tokens is None
            else issue146._safe_json_value(payload, host_tokens=host_tokens)
        )
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
