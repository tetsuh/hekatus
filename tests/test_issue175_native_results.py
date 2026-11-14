"""Device-free result framing, raw-stream and atomic-publication contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from enodia.tt.bench import issue120_results as results


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "OUTPUT_ROOT", tmp_path)
    return results.create_attempt("unit-run-001")


def execute_script(attempt, code, timeout=5):
    return results.execute(
        [sys.executable, "-c", code, str(attempt.result_file), attempt.identity],
        attempt, timeout, lambda: os.environ.copy(), attempt.directory,
    )


GOOD = """
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1])
os.write(1,b'prefix\\xff\\x00\\n')
os.write(2,b'error\\xfe\\x00\\n')
p.write_text(json.dumps({'schema':'issue120-native-result-v1','attempt_identity':sys.argv[2],'result':{'probe_id':'unit'}}))
os.write(1,b'cleanup after JSON\\n')
"""


def test_file_result_ignores_cleanup_logs_and_retains_exact_bytes(attempt):
    completed = execute_script(attempt, GOOD)
    assert completed.returncode == 0
    assert results.read_result(attempt) == {"probe_id": "unit"}
    assert attempt.stdout_path.read_bytes() == b"prefix\xff\x00\ncleanup after JSON\n"
    assert attempt.stderr_path.read_bytes() == b"error\xfe\x00\n"
    metadata = json.loads(attempt.execution_path.read_bytes())
    assert metadata["native_returncode"] == 0
    assert metadata["attempt_identity"] == attempt.identity
    assert metadata["native_started"] is True


@pytest.mark.parametrize("mode", ["missing", "two", "broken", "wrong_id", "wrong_schema", "wrong_payload", "wrong_root", "duplicate_key", "temp", "symlink"])
def test_result_rejection_keeps_streams_and_exit_code(attempt, mode):
    code = """
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); mode=MODE
os.write(1,b'raw\\xff');os.write(2,b'raw\\xfe')
e={'schema':'issue120-native-result-v1','attempt_identity':sys.argv[2],'result':{}}
if mode=='wrong_id':e['attempt_identity']='other-run'
if mode=='wrong_schema':e['schema']='other-schema'
if mode=='wrong_payload':e['result']=[]
if mode=='wrong_root':e=[]
if mode!='missing':p.write_text(json.dumps(e))
if mode=='two':(p.parent/'second.json').write_text('{}')
if mode=='temp':(p.parent/'result.json.tmp').write_text('{}')
if mode=='broken':p.write_bytes(b'{broken\\xff')
if mode=='duplicate_key':p.write_text('{"schema":"issue120-native-result-v1","schema":"issue120-native-result-v1"}')
if mode=='symlink':
 p.rename(p.parent.parent/'outside.json');p.symlink_to(p.parent.parent/'outside.json')
""".replace("MODE", repr(mode))
    assert execute_script(attempt, code).returncode == 0
    with pytest.raises(results.NativeResultError):
        results.read_result(attempt)
    assert attempt.stdout_path.read_bytes() == b"raw\xff"
    assert attempt.stderr_path.read_bytes() == b"raw\xfe"
    assert json.loads(attempt.execution_path.read_bytes())["native_returncode"] == 0


@pytest.mark.parametrize(("key", "value"), [("attempt_identity", "stale"), ("schema", "wrong"), ("native_returncode", False), ("native_started", 1), ("timed_out", 0)])
def test_execution_metadata_must_match_the_direct_capture(attempt, key, value):
    execute_script(attempt, GOOD)
    metadata = json.loads(attempt.execution_path.read_bytes())
    metadata[key] = value
    attempt.execution_path.write_text(json.dumps(metadata))
    with pytest.raises(results.NativeResultError):
        results.read_result(attempt)


def test_oversized_result_is_rejected_without_parsing(attempt):
    execute_script(attempt, GOOD)
    with attempt.result_file.open("wb") as handle:
        handle.truncate(results.MAX_RESULT_BYTES + 1)
    with pytest.raises(results.NativeResultError, match="bounded"):
        results.read_result(attempt)


def test_wrapper_record_reads_file_and_does_not_decode_native_stdout(attempt, monkeypatch):
    from enodia.tt.bench import issue120_probe as probe

    completed = execute_script(attempt, GOOD)
    completed.stdout = b"cleanup and invalid utf8 \\xff"
    monkeypatch.setattr(probe, "validate_probe_record", lambda _record: None)
    spec = probe.planned_probe("q1-cq1-completion", first_kernel_run=True)
    record = probe._record_native_result(completed, spec, {}, probe.PINNED_LIBRARY_HASH, True, attempt)
    assert record["probe_id"] == "unit"
    assert record["native_capture"]["attempt_identity"] == attempt.identity
    assert record["native_capture"]["native_returncode"] == 0


def test_unexpected_preflight_exception_still_records_capture(attempt):
    def broken():
        raise KeyError("unexpected fixture")

    with pytest.raises(KeyError):
        results.execute(["/not/launched"], attempt, 1, broken, attempt.directory)
    assert attempt.stdout_path.read_bytes() == b""
    assert attempt.stderr_path.read_bytes() == b""
    metadata = json.loads(attempt.execution_path.read_bytes())
    assert metadata["native_returncode"] is None
    assert metadata["failure_type"] == "KeyError"


def test_nonzero_native_cannot_be_success_even_with_valid_result(attempt):
    completed = execute_script(attempt, GOOD + "\nsys.exit(2)\n")
    assert completed.returncode == 2
    with pytest.raises(results.NativeResultError):
        results.read_result(attempt)
    assert attempt.stdout_path.read_bytes().endswith(b"cleanup after JSON\n")
    assert json.loads(attempt.execution_path.read_bytes())["native_returncode"] == 2


def test_timeout_saves_partial_bytes_and_direct_wait_status(attempt):
    code = "import os,time;os.write(1,b'partial\\xff');os.write(2,b'partial\\xfe');time.sleep(30)"
    with pytest.raises(results.NativeResultError, match="cap|timeout"):
        execute_script(attempt, code, timeout=1)
    assert attempt.stdout_path.read_bytes() == b"partial\xff"
    assert attempt.stderr_path.read_bytes() == b"partial\xfe"
    metadata = json.loads(attempt.execution_path.read_bytes())
    assert metadata["native_returncode"] == -9
    assert metadata["timed_out"] is True


def test_spawn_failure_keeps_empty_streams_and_does_not_infer_native_exit(attempt):
    with pytest.raises(results.NativeResultError):
        results.execute(["/missing/native"], attempt, 1, dict, attempt.directory)
    assert attempt.stdout_path.read_bytes() == b""
    assert attempt.stderr_path.read_bytes() == b""
    metadata = json.loads(attempt.execution_path.read_bytes())
    assert metadata["native_returncode"] is None
    assert metadata["native_started"] is False


def test_preflight_failure_has_no_native_exit_or_lost_capture(attempt):
    def rejected():
        raise ValueError("preflight fixture")

    with pytest.raises(results.NativeResultError):
        results.execute(["/not/launched"], attempt, 1, rejected, attempt.directory)
    assert attempt.stdout_path.read_bytes() == b""
    assert attempt.stderr_path.read_bytes() == b""
    assert json.loads(attempt.execution_path.read_bytes())["native_started"] is False


@pytest.mark.parametrize("identity", [None, "", "../escape", "a/b", "a;command", "a\nline", "a" * 129])
def test_bad_identity_never_creates_an_attempt(tmp_path, monkeypatch, identity):
    monkeypatch.setattr(results, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(results.NativeResultError):
        results.create_attempt(identity)
    assert not list(tmp_path.iterdir())


def test_attempt_never_reuses_a_prior_directory(attempt):
    attempt.result_file.write_bytes(b"immutable")
    with pytest.raises(results.NativeResultError):
        results.create_attempt(attempt.identity)
    assert attempt.result_file.read_bytes() == b"immutable"


def test_output_root_redirect_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "root"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(results, "OUTPUT_ROOT", link)
    with pytest.raises(results.NativeResultError):
        results.create_attempt("valid-run")
    assert not list(target.iterdir())


@pytest.fixture(scope="module")
def writer(tmp_path_factory):
    directory = tmp_path_factory.mktemp("result-writer")
    source = directory / "main.cpp"
    source.write_text('''
#include "native_result_io.hpp"
#include <iostream>
int main(int argc, char** argv) {
    try { issue120::publish_result(argv[1], argv[2], argv[3], argv[4]); return 0; }
    catch (const std::exception& e) { std::cerr << e.what(); return 2; }
}
''')
    binary = directory / "cpu-result-writer"
    subprocess.run(["g++", "-std=c++20", "-I", "enodia/tt/bench/issue120", str(source), "-o", str(binary)], check=True)
    return binary


def run_writer(writer, root, result_file, identity="unit-run"):
    return subprocess.run([str(writer), str(root), str(result_file), identity, '{"probe_id":"unit"}'], capture_output=True, check=False)


def writer_target(tmp_path):
    target = tmp_path / "native-unit-run" / "result" / "result.json"
    target.parent.mkdir(parents=True)
    return target


def test_cpp_publication_is_atomic_schema_bound_and_non_overwriting(writer, tmp_path):
    target = writer_target(tmp_path)
    assert run_writer(writer, tmp_path, target).returncode == 0
    value = json.loads(target.read_bytes())
    assert value["schema"] == "issue120-native-result-v1"
    assert value["attempt_identity"] == "unit-run"
    assert value["result"] == {"probe_id": "unit"}
    assert list(target.parent.iterdir()) == [target]
    before = target.read_bytes()
    assert run_writer(writer, tmp_path, target).returncode == 2
    assert target.read_bytes() == before


@pytest.mark.parametrize("mode", ["temp_exists", "wrong_path", "bad_identity", "parent_link"])
def test_cpp_rejects_non_owned_or_stale_target(writer, tmp_path, mode):
    target = writer_target(tmp_path)
    identity = "unit-run"
    if mode == "temp_exists":
        target.with_suffix('.json.tmp').write_bytes(b"stale")
    elif mode == "wrong_path":
        target = tmp_path / "arbitrary.json"
    elif mode == "bad_identity":
        identity = "../escape"
    else:
        target.parent.rmdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target.parent.symlink_to(outside, target_is_directory=True)
    assert run_writer(writer, tmp_path, target, identity).returncode == 2
    assert not target.exists()


def test_cpp_main_validates_protocol_before_opening_a_mesh():
    source = Path('enodia/tt/bench/issue120/issue120_probe.cpp').read_text()
    main = source[source.index('int main('):]
    assert main.index('validate_result_target') < main.index('run_probe(args)')
    assert 'publish_result' in main
    assert 'std::cout << result.dump()' not in main
