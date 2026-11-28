"""Issue #30 fixed-layout, attempt-bound Ethernet capture and judgments.

Planning/import is board-free. Capture defaults to an isolated no-device image;
--approved-board additionally requires separate owner files at fixed slots.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from enodia.tt.bench import issue30_admission as admission
from enodia.tt.bench import issue30_diagnostics as diagnostics
from enodia.tt.bench import issue30_evidence as evidence
from enodia.tt.bench import issue30_formats as formats
from enodia.tt.bench import issue30_topology as topology

IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "out/issue30"
ASSETS = WORK / "g0/assets"
SOURCE_ROOT = WORK / "tt-metal"
CURRENT_RECEIPT = ROOT / "docs/measurements/issue30-g0/receipt-v16.json"
DOCKER_SOCKET = "/var/run/docker.sock"
DOCKER_CLI = ("docker", "--host", "unix://" + DOCKER_SOCKET)
OWNER_ACCEPTANCE = "docs/measurements/2026-11-24-issue30-a-owner-acceptance/accepted-a.json"
SOURCE_COMMIT = "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd"
UMD_COMMIT = "bc77ac2f7382ffeb9fa5430055f0854d53979489"
LIBRARY_SHA256 = "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141"
CARDS = (("000004033191411c", "0000:06:00.0"), ("0000040331914124", "0000:09:00.0"))
TESTS = {"A": "Cluster.ReportSystemHealth", "B": "Fabric1DFixture.TestUnicastConnAPI1DSmoke"}
BINARIES = {"A": "test_system_health", "B": "fabric_smoke_tests"}
ACCEPTED = {"A": "accepted_discovery", "B": "accepted_transfer"}
SCOPES = {"A": "one discovery", "B": "five checked packets; no timing claim"}
DEADLINES = {"binary": 40, "container": 50, "cleanup": 60}
GOOD_EXECUTION = {"exit": 0, "client_exit": 0, "started": True, "elapsed_seconds": 1.0,
                  "timed_out": False, "terminated": True,
                  "watcher_clean": True, "containers_clean": True}
CONTAINERS = "containers/"
DOWN = "DOWN/unconnected"
EXECUTION_FILE = "execution.json"
STDOUT_FILE = "stdout.bin"
STDERR_FILE = "stderr.bin"
BEFORE_FILE = "before.json"
MASKS = ["/sys", "/proc/bus", "/proc/driver", "/proc/acpi", "/proc/asound",
         "/proc/interrupts", "/proc/kcore", "/proc/keys", "/proc/latency_stats",
         "/proc/timer_list", "/proc/timer_stats", "/proc/sched_debug", "/proc/scsi"]
PROGRAM_FILES = ("enodia/tt/bench/issue30.py", "enodia/tt/bench/issue30_build/CMakeLists.txt",
                 "tools/build_issue30.py", "tests/test_issue30.py", "tests/test_issue30_capture.py",
                 "pyproject.toml", "uv.lock", "enodia/tt/bench/issue30_topology.py",
                 "tests/test_issue30_topology.py", topology.POLICY_FILE,
                 "enodia/tt/bench/issue30_admission.py", "tests/test_issue30_admission.py",
                 "docs/adr/0021-discovery-observability-gate.md", OWNER_ACCEPTANCE,
                 "enodia/tt/bench/issue30_diagnostics.py", "tests/test_issue30_owner.py",
                 "enodia/tt/bench/issue120_production/checks.py",
                 "docs/measurements/2026-11-14-p150a-issue120-p6-api-lock-semaphore-v3-baseline-verification.json",
                 "docs/measurements/2026-11-24-issue30-a-discovery/A-discovery.json",
                 "docs/measurements/2026-11-24-issue30-a-discovery/A-native-evidence.tar.gz",
                 "docs/measurements/2026-11-24-issue30-a-owner-acceptance/baseline-match.json",
                 "enodia/tt/bench/issue30_runtime.py", "enodia/tt/bench/issue120_runtime.py",
                 "enodia/tt/bench/issue196_compile.py", "enodia/tt/bench/issue196_compile_manifest.json",
                 "tools/check_issue196_device_kernels.py", "tests/test_issue196.py",
                 "docs/measurements/issue196/offline-inputs.tar.gz",
                 "docs/measurements/issue196/official-smoke-source.txt",
                 "docs/measurements/issue196/morning-admission.json",
                 "tests/test_issue30_morning.py",
                 "tests/test_issue30_dram.py",
                 "docs/measurements/2026-11-24-issue30-b-smoke/B-native-evidence.tar.gz",
                 "enodia/tt/bench/issue30_evidence.py", "tests/test_issue30_evidence.py",
                 "tests/test_issue30_raw_evidence.py", "tools/replay_issue30_evidence.py",
                 "tests/test_issue30_replay_paths.py", "tests/test_issue30_final_binding.py",
                 "enodia/tt/bench/issue30_formats.py", "tests/test_issue200_201.py",
                 "tests/test_issue200_201_loop1.py", "tests/test_issue200_201_loop2.py",
                 "tests/test_issue202_final_fix.py",
                 "tests/test_issue200_lifecycle.py", "tests/test_issue201_sweep.py",
                 "docs/adr/0022-issue30-incomplete-offline-audit.md")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def program_sources() -> dict:
    return {name: digest((ROOT / name).read_bytes()) for name in PROGRAM_FILES}


def _identity(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", value):
        raise ValueError("unsafe attempt identity")
    return value


def _owned(path: Path, base: Path | None = None) -> Path:
    root = (base or WORK).absolute()
    candidate = path.absolute()
    if root.resolve() != root or candidate.resolve() != candidate or not candidate.is_relative_to(root):
        raise ValueError("path is not canonical program-owned storage")
    return candidate


def _read(path: Path, base: Path | None = None, limit: int = 8 * 1024 * 1024) -> bytes:
    selected = _owned(path, base)
    if selected.is_symlink() or not selected.is_file() or selected.stat().st_size > limit:
        raise ValueError("missing, redirected or oversized evidence")
    return selected.read_bytes()


def artifact(path: Path, root: Path) -> dict:
    selected = _owned(path, root)
    if selected.is_symlink():
        raise ValueError("redirected artifact")
    hashed, count = hashlib.sha256(), 0
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            hashed.update(block)
    return {"path": selected.relative_to(root.absolute()).as_posix(),
            "bytes": count, "sha256": hashed.hexdigest()}


def _verified(path, root, expected):
    """Read once, then verify before parsing or consuming evidence."""
    try:
        raw = _read(path, root, limit=diagnostics.checks.MAX_WATCHER_BYTES)
    except (OSError, ValueError) as exc:
        raise ValueError("capture_artifact") from exc
    if (expected.get("sha256") != digest(raw)
            or "bytes" in expected and (type(expected["bytes"]) is not int or expected["bytes"] != len(raw))):
        raise ValueError("capture_artifact")
    return raw


def _captured(output, execution, name):
    entries = execution.get("artifacts", [])
    formats.manifest(entries)
    matches = [entry for entry in entries if entry.get("path") == name]
    if len(matches) != 1 or set(matches[0]) != {"path", "bytes", "sha256"}:
        raise ValueError("capture_artifact")
    return _verified(output / name, output, matches[0])


def _native(output, execution):
    return [_captured(output, execution, name)
            for name in ("native/gtest.xml", STDOUT_FILE, STDERR_FILE)]


def publish(path: Path, record: dict) -> None:
    """Linux no-replace atomic files; no symlink/overwrite fallback."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path.name)
    if path.absolute() != path.resolve():
        raise ValueError("redirected publication")
    temporary = path.with_name(path.name + ".tmp")
    owned, renamed = False, False
    try:
        with temporary.open("xb") as output:
            owned = True
            output.write((json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        _rename(temporary, path)
        renamed = True
    finally:
        if owned and not renamed:
            temporary.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _rename(temporary, path):
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temporary), -100, os.fsencode(path), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def remaining(start: float, limit: int, now: float) -> float:
    return max(0.0, start + limit - now)


def request(phase: str, identity: str, assets: Path, capture: Path, *, uid: int, gid: int) -> dict:
    _identity(identity)
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid < 0:
        raise ValueError("non-root identity required")
    scratch = f"rw,nosuid,nodev,noexec,size=256m,mode=0700,uid={uid},gid={gid}"
    return {"Image": IMAGE, "User": f"{uid}:{gid}", "WorkingDir": "/capture",
            "Env": ["TT_METAL_WATCHER=1", "HOME=/scratch", "LD_BIND_NOW=1", "TMPDIR=/scratch",
                    "TT_METAL_RUNTIME_ROOT=/capture/runtime", "TT_METAL_KERNEL_PATH=/source",
                    "TT_METAL_CACHE=/capture/cache", "TT_METAL_LOGS_PATH=/capture/logs"],
            "Cmd": ["timeout", "--signal=KILL", "40s", "python3",
                    "/harness/enodia/tt/bench/issue30_runtime.py", phase],
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Devices": [],
                           "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                           "MaskedPaths": MASKS, "Tmpfs": {"/scratch": scratch},
                           "Binds": [f"{assets.resolve()}:/assets:ro", f"{capture.resolve()}:/capture:rw",
                                     f"{ROOT}:/harness:ro", f"{SOURCE_ROOT}:/source:ro"]}}


def approval_record(phase, identity, env, receipt, head):
    record = {"schema": "issue30-owner-approval-v1", "attempt": identity, "phase": phase,
              "head": head, "boot": env["boot"],
              "environment_sha256": digest(json.dumps(env, sort_keys=True).encode()),
              "receipt_sha256": digest(json.dumps(receipt, sort_keys=True).encode())}
    if "owner_unit" in env:
        unit = _renewed_unit(env)
        if identity != unit[phase]:
            raise ValueError("owner_unit_identity")
        return record | {"schema": "issue30-owner-approval-v4", "owner_unit": unit,
                         "owner_authorization": {"date": unit["date"], "invocations": 1,
                                                 "test": TESTS[phase], "watcher": "1",
                                                 "cards": [list(c) for c in CARDS], "max_seconds": 60},
                         "conditional_B": "accepted fresh H1 A, identical boot and cabling"}
    if phase == "A":
        record |= {"schema": "issue30-owner-approval-v2", "operator_correction": dict(admission.CORRECTION),
                   "owner_authorization": {"date": "2026-11-24", "logical_unit": "issue30-a-discovery-watcher-v1",
                                           "invocations": 1, "test": TESTS["A"], "watcher": "1",
                                           "cards": [list(c) for c in CARDS], "max_seconds": 60}}
    if phase == "B":
        record |= {"schema": "issue30-owner-approval-v3",
                   "accepted_A_sha256": digest((ROOT / OWNER_ACCEPTANCE).read_bytes()),
                   "owner_decision": {"date": "2026-11-24", "logical_unit": "issue30-b-fabric1d-smoke-watcher-v1",
                                      "invocations": 1, "test": TESTS["B"], "watcher": "1",
                                      "cards": [list(c) for c in CARDS], "max_seconds": 60,
                                      "cabling_unchanged_since_A": True, "no_A_recheck": True}}
    return record


def _approval(phase, identity, approval, env, receipt, head):
    expected = approval_record(phase, identity, env, receipt, head)
    if not formats.equal(approval, expected):
        raise ValueError("approval_binding")
    if (type(receipt) is not dict or receipt.get("schema") != "issue30-g0-v1"
            or receipt.get("all_passed") is not True):
        raise ValueError("g0_not_ready")
    if not formats.equal(receipt.get("current_wrapper_sources"), program_sources()):
        raise ValueError("receipt_source_binding")


def _binary_binding(assets, receipt):
    if type(receipt) is not dict or type(receipt.get('binaries')) is not dict:
        raise ValueError('binary_binding')
    for name in BINARIES.values():
        path = assets / name
        if path.is_symlink() or digest(path.read_bytes()) != receipt.get("binaries", {}).get(name):
            raise ValueError("binary_binding")


def board_request(phase: str, identity: str, assets: Path, output: Path, *,
                  approval: dict, environment: dict, receipt: dict, head: str) -> dict:
    _environment(environment, identity, phase)
    _approval(phase, identity, approval, environment, receipt, head)
    _binary_binding(assets, receipt)
    payload = request(phase, identity, assets, output, uid=os.getuid(), gid=os.getgid())
    nodes = _approved_nodes(environment)
    host = payload["HostConfig"]
    host["Devices"] = [{"PathOnHost": n, "PathInContainer": n, "CgroupPermissions": "rw"} for n in nodes]
    host["MaskedPaths"] = [p for p in MASKS if p not in ("/sys", "/proc/bus", "/proc/driver")]
    host["Binds"] += ["/dev/hugepages-1G:/dev/hugepages-1G:rw"]
    host["Ulimits"] = [{"Name": "memlock", "Soft": -1, "Hard": -1}]
    payload["Env"] += ["TT_VISIBLE_DEVICES=" + ",".join(c[1] for c in CARDS)]
    return payload


def _approved_nodes(environment):
    nodes = [c.get("node", "") for c in environment["cards"]]
    if (any(type(n) is not str or not re.fullmatch(r"/dev/tenstorrent/\d{1,6}", n, re.ASCII) for n in nodes)
            or len(set(nodes)) != 2):
        raise ValueError("device_mapping")
    for serial, pci in CARDS:
        matches = [c for c in environment['cards'] if (c.get('serial'), c.get('pci')) == (serial, pci)]
        if len(matches) != 1:
            raise ValueError('device_pci_binding')
        # Only numeric data crosses into the fixed sysfs class namespace. Keep
        # display padding, so this construction preserves the admitted spelling.
        component = matches[0]['node'].rsplit('/', 1)[1]
        index, width = int(component), len(component)
        associated = Path('/sys/class/tenstorrent') / f'tenstorrent!{index:0{width}d}' / 'device'
        expected = Path('/sys/bus/pci/devices') / pci  # From CARDS, never file content.
        try:
            if associated.resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError("device_pci_binding")
        except OSError as exc:
            raise ValueError("device_pci_binding") from exc
    return nodes


def _bundle_report_name(value):
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}(?:\.0)?", value, re.ASCII):
        raise ValueError("environment")
    return ".".join(str(int(x)) for x in value.split(".")[:3])


def _card(card, phase):
    cmfw = card.get("cmfw", "")
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?", cmfw, re.ASCII):
        raise ValueError("environment")
    if not any(map(int, cmfw.split("."))):
        raise ValueError("environment")
    if tuple(map(int, _bundle_report_name(card.get("bundle", "")).split("."))) < (18, 5, 0):
        raise ValueError("environment")
    admission.card_health(card, phase)
    if card.get("dram_healthy") is not True:
        raise ValueError("environment")
    if type(card.get("aiclk_mhz")) is not int or card["aiclk_mhz"] != 800 or type(card.get("pcie_index")) is not int:
        raise ValueError("environment")


def _cards(cards, phase, require_arc):
    if not isinstance(cards, list) or len(cards) != 2:
        raise ValueError("environment")
    if {c.get("chip") for c in cards} != {0, 1}:
        raise ValueError("environment")
    if {(c.get("serial"), c.get("pci")) for c in cards} != set(CARDS):
        raise ValueError("environment")
    for c in cards:
        _card(c, phase)
        if require_arc or 'arc_heartbeat' in c:
            admission.arc_progress(c)
    if len({c["pcie_index"] for c in cards}) != 2 or len({c["bundle"] for c in cards}) != 1:
        raise ValueError("environment")


def _links(env):
    if "topology" in env:
        topology.validate_input(env)
        return
    links = env.get("links", [])
    keys = {(x["chip"], x["channel"]) for x in links}
    if not links or len(keys) != len(links) or keys & {tuple(x) for x in env.get("uncabled", [])}:
        raise ValueError("environment")
    for link in links:
        reverse = any(x["chip"] == link["peer"] and x["core"] == link["peer_core"]
                      and x["peer_core"] == link["core"] for x in links)
        if link["chip"] not in (0, 1) or link["peer"] != 1 - link["chip"] or not reverse:
            raise ValueError("environment")


def _environment(env, identity, phase="B"):
    formats.environment(env)
    if (env.get("schema") != "issue30-environment-v1" or env.get("attempt") != identity
            or not isinstance(env.get("boot"), str) or not env["boot"]
            or env.get("image") != IMAGE or env.get("watcher") != "1"):
        raise ValueError("environment")
    kmd = env.get("kmd", "")
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}", kmd, re.ASCII) or tuple(map(int, kmd.split("."))) < (2, 8, 0):
        raise ValueError("environment")
    _cards(env.get("cards"), phase, "topology" in env)
    admission.dram_gate(env)
    if 'owner_unit' in env and env['attempt'] != _renewed_unit(env)[phase]:
        raise ValueError('owner_unit_identity')
    _links(env)


def _xml(phase, raw):
    formats.xml(raw, TESTS[phase])


def _strip_source(text):
    body, sep, tail = text.rpartition(" (")
    filename, colon, line = tail.removesuffix(")").rpartition(":")
    if sep and tail.endswith(")") and colon and re.fullmatch(r'\d+', line, re.ASCII) and len(filename) <= 256:
        return body
    return text


def _message(line):
    envelope = formats.logger(line)
    if envelope is not None:
        return envelope['level'].strip(), _strip_source(envelope['body'])
    if line.count('|') >= 3:
        raise ValueError('diagnostic')
    return "", line.strip()


def _chip(text):
    match = re.fullmatch(r'Chip: (\d{1,10}) PCIe: (\d{1,10}) Unique ID: ([0-9a-f]{1,16})', text, re.ASCII)
    if match is None:
        raise ValueError('chip_identity')
    return int(match[1]), int(match[2]), match[3]


def _link(text):
    core = r'\d{1,3}-\d{1,3}'
    match = re.fullmatch(r'eth channel (\d{1,10}) core (' + core + r') link (.+)', text, re.ASCII)
    if match is None:
        raise ValueError('peer_links')
    data = {'channel': int(match[1]), 'core': match[2], 'state': DOWN}
    if match[3] == DOWN + ' (QSFP)':
        return data
    peer = re.fullmatch(r'UP \(QSFP\), connected to (Chip \d{1,10}|Unique ID: [a-f0-9]{1,16}) core (' + core + ')', match[3], re.ASCII)
    if peer is None:
        raise ValueError('peer_links')
    key, value = ('peer', int(peer[1][5:])) if peer[1].startswith('Chip ') else ('peer_unique_id', peer[1][11:])
    return data | {'state': 'UP', key: value, 'peer_core': peer[2]}


def _warning(env, body, line):
    header, sep, link = body.partition(" eth channel ")
    try:
        chip, pcie, uid = _chip(header)
        parsed = _link("eth channel " + link)
    except ValueError as exc:
        raise ValueError("diagnostic") from exc
    indices = {c["chip"]: c["pcie_index"] for c in env["cards"]}
    allowed = ([chip, parsed["channel"]] in env.get("uncabled", [])
               or "topology" in env and parsed["channel"] in topology.CHANNEL_PORT)
    if not sep or parsed["state"] != DOWN or indices.get(chip) != pcie or not allowed:
        raise ValueError("diagnostic")
    envelope = formats.logger(line)
    if (envelope is None or envelope['component'] != 'Test' or envelope['level'] != 'warning  '
            or not line.endswith(' (test_system_health.cpp:361)')):
        raise ValueError("diagnostic")
    return {"chip": chip, "unique_id": uid} | parsed


def _diagnostic_lines(raw):
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("diagnostic")
    try:
        text = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("diagnostic") from exc
    return text.splitlines()


def _down_envelope(body, level, transfer):
    down = re.search(r'\bDOWN\b', body, re.IGNORECASE)
    summary = re.search(r'\b(?:Chip|PCIe)[ \t]*:', body, re.IGNORECASE)
    if transfer:
        summary = summary or re.search(r'\beth[ \t]+channel\b', body, re.IGNORECASE)
    if down and summary and level != 'warning':
        raise ValueError('diagnostic')


def _messages(env, raw, warnings=None, accepted=None, uids=None, *, transfer=False):
    messages = []
    lines = _diagnostic_lines(raw)
    for index, line in enumerate(lines):
        if diagnostics.SERIOUS.search(line):
            raise ValueError("diagnostic")
        if diagnostics.baseline_at(lines, index, env["cards"]):
            continue
        level, body = _message(line)
        _health_line(env, line, body)
        _down_envelope(body, level, transfer)
        if level == "warning":
            warning = _warning(env, body, line)
            _accepted_warning(accepted, uids, warning)
            if warnings is not None:
                warnings.append(warning)
        elif diagnostics.ALARM.search(line):
            raise ValueError("diagnostic")
        else:
            messages.append(body)
    return messages


def _accepted_warning(records, uids, warning):
    if records is None:
        return
    topology.check_warnings(records, [warning])
    if uids is None or uids.get(warning['chip']) != warning['unique_id']:
        raise ValueError('diagnostic')


def _health_line(env, line, body):
    bundle = 'Established firmware bundle version: '
    if body.startswith(bundle):
        envelope = formats.logger(line)
        expected = bundle + _bundle_report_name(env['cards'][0]['bundle'])
        if (envelope is None or envelope['level'] != 'info     ' or envelope['component'] != 'UMD'
                or envelope['body'] != expected + ' (topology_discovery.cpp:574)'):
            raise ValueError('bundle')
    if body.startswith('Found '):
        envelope = formats.logger(line)
        if envelope is None or envelope['level'] != 'info     ' or envelope['component'] != 'Test':
            raise ValueError('chip_count')


def _observed_link(lines, index, chip):
    data = _link(lines[index])
    if chip is None:
        raise ValueError("peer_links")
    if data["state"] == "UP":
        if index + 1 >= len(lines) or lines[index + 1] != "Retrain count: 0":
            raise ValueError("retrain")
        data["retrain_count"] = 0
    return {"chip": chip} | data


def _legacy_links(env, observed):
    found = []
    for data in observed:
        if data["state"] == DOWN:
            if [data["chip"], data["channel"]] not in env.get("uncabled", []):
                raise ValueError("peer_links")
        else:
            found.append({k: v for k, v in data.items() if k not in ("state", "retrain_count", "unique_id")})
    canonical = lambda xs: sorted(json.dumps(x, sort_keys=True) for x in xs)
    if canonical(found) != canonical(env["links"]):
        raise ValueError("peer_links")


def _health_records(env, lines, observed, warnings):
    headers, chip = [], None
    for index, body in enumerate(lines):
        if body.startswith("Chip: "):
            chip, pcie, uid = _chip(body)
            headers.append((chip, pcie, uid))
        elif body.startswith("eth channel "):
            observed.append(_observed_link(lines, index, chip))
    _checked_uids(env, headers, warnings)
    if "topology" in env:
        topology.check_warnings(observed, warnings)
        return topology.classify(observed)
    _legacy_links(env, observed)
    return None


def _checked_uids(env, headers, warnings=()):
    expected = {(c['chip'], c['pcie_index']) for c in env['cards']}
    if len(headers) != 2 or {(c, p) for c, p, _ in headers} != expected:
        raise ValueError('chip_identity')
    uids = {c: uid for c, _, uid in headers}
    if len(set(uids.values())) != 2 or any(w['unique_id'] != uids.get(w['chip']) for w in warnings):
        raise ValueError('diagnostic')
    return uids


def _health(env, raw, observed):
    warnings = []
    lines = _messages(env, raw, warnings)
    if [x for x in lines if x.startswith("Found ")] != ["Found 2 chips in cluster:"]:
        raise ValueError("chip_count")
    prefix = "Established firmware bundle version: "
    bundles = [x.removeprefix(prefix) for x in lines if x.startswith(prefix)]
    if bundles != [_bundle_report_name(env["cards"][0]["bundle"])]:
        raise ValueError("bundle")
    decision = _health_records(env, lines, observed, warnings)
    if "owner_unit" in env and decision.get("hypothesis") != "H1":
        raise ValueError("fresh_A_H1_required")
    return decision


def baseline_digest(env):
    return digest(json.dumps(admission.stable_environment(env), sort_keys=True).encode())


def _execution(execution, phase="B"):
    if (type(execution) is not dict or type(execution.get("exit")) is not int or execution["exit"] != 0
            or execution.get("timed_out") is not False or execution.get("terminated") is not True
            or execution.get("started") is not True
            or type(execution.get("client_exit")) is not int or execution["client_exit"] != 0):
        raise ValueError("execution")
    elapsed = execution.get("elapsed_seconds")
    if not formats.finite(elapsed) or not 0 <= elapsed <= 60:
        raise ValueError("execution")
    if phase == "B" and execution.get("watcher_clean") is not True:
        raise ValueError("watcher")
    if execution.get("containers_clean") is not True:
        raise ValueError("containers")
    if 'error_type' in execution or execution.get('create_status', 'identified') != 'identified':
        raise ValueError('execution')


def _post(env, after, identity, phase):
    _environment(env, identity, phase)
    _environment(after, identity, phase)
    if baseline_digest(env) != baseline_digest(after):
        raise ValueError("environment")


def _transfer_binding(before, discovery, raw, uids=None):
    if not discovery:
        raise ValueError("discovery_binding")
    if "topology" in before:
        if "owner_unit" in before:
            _fresh_links(before, discovery)
        else:
            admission.owner_links(discovery, baseline_digest(before), digest(before["boot"].encode()),
                                  digest(json.dumps(before["topology"], sort_keys=True).encode()))
        _messages(before, raw, accepted=discovery["observed_qsfp"], uids=uids, transfer=True)
    else:  # Historical CPU fixtures; hardware CLI always requires the frozen topology.
        if (discovery.get("phase") != "A" or discovery.get("verdict") != ACCEPTED["A"]
                or discovery.get("baseline_sha256") != baseline_digest(before)):
            raise ValueError("discovery_binding")
        _messages(before, raw, transfer=True)


def _marker_position(lines, bodies, marker):
    found = [i for i, body in enumerate(bodies) if body == marker
             and (lines[i] == marker if marker.startswith("[") else _message(lines[i])[0] == "info")]
    if len(found) != 1 or sum(marker in line for line in lines) != 1:
        raise ValueError("packet_evidence")
    return found[0]


def _packet_observation(before, raw, stderr=b""):
    if "topology" not in before:
        return None
    lines = _diagnostic_lines(raw)
    if any(re.search(r"\[\s*(?:RUN|OK|PASSED|FAILED|SKIPPED)\s*\]|Running fabric api tests|Fabric Reliability Mode:",
                     line) for line in _diagnostic_lines(stderr)):
        raise ValueError("packet_evidence")
    bodies = [_message(line)[1] for line in lines]
    required = ("Running fabric api tests with fast dispatch",
                "Fabric Reliability Mode: STRICT_SYSTEM_HEALTH_SETUP_MODE",
                "[ RUN      ] " + TESTS["B"], "[  PASSED  ] 1 test.")
    positions = [_marker_position(lines, bodies, marker) for marker in required]
    native = [line for line in lines if re.search(r'\[\s*(?:RUN|OK|PASSED|FAILED|SKIPPED)\s*\]', line)]
    if len(native) != 3:
        raise ValueError('packet_evidence')
    for index, location in zip(positions[:2], ('87', '100'), strict=True):
        envelope = formats.logger(lines[index])
        if (envelope is None or envelope['component'] != 'Test'
                or envelope['body'] != bodies[index] + ' (fabric_fixture.hpp:' + location + ')'):
            raise ValueError("packet_evidence")
    completed = [i for i, line in enumerate(lines) if re.fullmatch(
        r"\[       OK \] " + re.escape(TESTS["B"]) + r" \(\d+ ms\)", line, re.ASCII)]
    if (len(completed) != 1 or sum("[       OK ] " + TESTS["B"] in line for line in lines) != 1
            or sum("Running fabric api tests" in line for line in lines) != 1
            or sum("Fabric Reliability Mode:" in line for line in lines) != 1
            or int(lines[completed[0]].rsplit('(', 1)[1].split()[0]) > 40000
            or not positions[0] < positions[1] < positions[2] < completed[0] < positions[3]):
        raise ValueError("packet_evidence")
    return {"expected_packets_from_pinned_source": 5,
            "basis": "source_bound_official_receiver_assertions",
            "payload_bytes_or_status_words_printed": False,
            "scope": "nonbenchmark five-packet PRNG receiver checks and sender/receiver status/byte equality; not timing"}


def _renewed_unit(env):
    unit = env.get("owner_unit", {})
    if set(unit) != {"date", "A", "B"} or unit.get("date") != "2026-11-25":
        raise ValueError("owner_unit")
    for phase, prefix in (("A", "issue30-a-discovery-watcher-v2"),
                          ("B", "issue30-b-fabric1d-smoke-watcher-v2")):
        if not _identity(unit[phase]).startswith(prefix + "-"):
            raise ValueError("owner_unit_identity")
    return unit


def _fresh_links(before, record):
    formats.verdict(record)
    unit = _renewed_unit(before)
    decision = topology.classify(record.get("observed_qsfp"))
    if (record.get("schema") != "issue30-verdict-v3" or record.get("phase") != "A"
            or record.get("attempt") != unit["A"]
            or record.get("verdict") != ACCEPTED["A"] or record.get("reason") != "accepted"
            or record.get("baseline_sha256") != baseline_digest(before)
            or not formats.equal(decision, record.get("topology")) or decision.get("hypothesis") != "H1"
            or decision.get("outcome") != "accepted"):
        raise ValueError("fresh_A_binding")


def _fresh_evidence(before, record):
    unit = _renewed_unit(before)
    output, inputs = _slots(unit["A"], True)
    original = _load(output / "result.json")
    checked = _judgment("A", unit["A"], output, inputs)
    _fresh_result(before, record, original, checked)


def _fresh_result(before, record, original, checked):
    if not formats.equal(record, original) or record.get('attempt') != _renewed_unit(before)['A']:
        raise ValueError('fresh_A_canonical_result')
    if not formats.equal(checked, original):
        raise ValueError('fresh_A_revalidation')
    _fresh_links(before, record)


def _fresh_native(before, record):
    unit = _renewed_unit(before)
    _, inputs = _slots(unit["A"], True)
    first = _load(inputs / BEFORE_FILE)
    _approval("A", unit["A"], _load(inputs / "approval.json"), first,
              _load(CURRENT_RECEIPT, ROOT), _git(ROOT, "rev-parse", "HEAD"))
    _fresh_evidence(before, record)


def _bound_discovery(inputs):
    record = _load(inputs / "discovery.json")
    before = _load(inputs / BEFORE_FILE)
    if "owner_unit" in before:
        _fresh_native(before, record)
        return record
    expected = _load(ROOT / OWNER_ACCEPTANCE, ROOT)
    if not formats.equal(record, expected):
        raise ValueError("owner_acceptance_binding")
    return record


def evaluate(phase, identity, before, after, xml, stdout, stderr, execution, discovery=None,
             discovery_uids=None):
    reason, decision, observed, packets = "accepted", None, [], None
    firmware = admission.observe_firmware(stdout + b"\n" + stderr)
    try:
        _execution(execution, phase)
        _xml(phase, xml)
        _post(before, after, identity, phase)
        # Validate independent streams before any combined topology interpretation.
        _messages(before, stdout)
        _messages(before, stderr)
        if phase == "A":
            decision = _health(before, stdout + b"\n" + stderr, observed)
            if decision is not None:
                reason = decision["reason"]
        else:
            _transfer_binding(before, discovery, stdout + b"\n" + stderr, discovery_uids)
            packets = _packet_observation(before, stdout, stderr)
        admission.check_reported_versions(firmware)
    except (ValueError, KeyError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "environment"
    result = {"schema": "issue30-verdict-v1", "attempt": identity, "phase": phase,
              "verdict": ACCEPTED[phase] if reason == "accepted" else "rejected", "reason": reason,
              "scope": SCOPES[phase], "baseline_sha256": baseline_digest(before),
              "evidence_sha256": {"xml": digest(xml), "stdout": digest(stdout), "stderr": digest(stderr)}}
    if type(before) is dict and "topology" in before:
        result |= {"schema": "issue30-verdict-v3", "topology": decision, "observed_qsfp": observed,
                   "firmware_observation": firmware, "packet_observation": packets}
        if decision is not None and decision["reason"] == reason:
            result["verdict"] = {"accepted": ACCEPTED[phase], "not_established": "link_not_established",
                                 "unexpected": "unexpected_topology"}.get(decision["outcome"], "rejected")
    return result


def _api(method, path, body=None, timeout=5, *, before_request=None):
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker API budget", timeout)
    command = ["curl", "-fsS", "--noproxy", "*", "--max-time", str(timeout), "--unix-socket",
               DOCKER_SOCKET, "-X", method, "-H", "Content-Type: application/json"]
    if body is not None:
        command += ["--data-binary", "@-"]
    command += ["http://localhost/v1.52/" + path]
    encoded = json.dumps(body).encode() if body is not None else None
    if before_request is not None:
        before_request()
    done = subprocess.run(command, input=encoded,
                          capture_output=True, timeout=timeout + 0.1, check=True)
    return formats.load(done.stdout) if done.stdout else {}


def _docker_ps(path, timeout):
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker ps budget", timeout)
    done = subprocess.run([*DOCKER_CLI, "ps", "-a", "--format", "{{.Names}}\t{{.State}}\t{{.Image}}"],
                          capture_output=True, check=True, timeout=timeout, env=_docker_environment())
    with path.open("xb") as stream:
        stream.write(done.stdout)


def _docker_environment():
    return {k: v for k, v in os.environ.items() if not k.startswith('DOCKER_')}


def _container_id(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid daemon container identity")
    return value


def _run_client(_name, cid, output, start, state, *, before_start=None):
    with (output / STDOUT_FILE).open("xb") as stdout, (output / STDERR_FILE).open("xb") as stderr:
        command = [*DOCKER_CLI, 'start', '-a', '--', _container_id(cid)]
        environment = _docker_environment()
        now = before_start() if before_start is not None else time.monotonic()
        if remaining(start, 50, now) <= 0:
            raise subprocess.TimeoutExpired('docker start budget', 0)
        client = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        state["started"] = True
        try:
            state["client_exit"] = client.wait(timeout=remaining(start, 50, time.monotonic()))
        except subprocess.TimeoutExpired:
            state["timed_out"] = True
            try:
                _api("POST", CONTAINERS + cid + "/kill", timeout=remaining(start, 55, time.monotonic()))
            finally:
                client.kill()
                client.wait(timeout=remaining(start, 57, time.monotonic()))
    observed = _api("GET", CONTAINERS + cid + "/json", timeout=remaining(start, 58, time.monotonic()))
    state["exit"] = observed["State"]["ExitCode"]
    state["terminated"] = observed["State"]["Running"] is False
    state["inner_deadline_status"] = {124: "timeout_exit", 137: "sigkill_or_deadline_unidentified"}.get(state["exit"], "not_indicated")
    state["timed_out"] |= state["exit"] == 124


def _cleanup(_name, cid, output, start, state):
    try:
        if cid is not None:
            _api("DELETE", CONTAINERS + cid + "?force=1", timeout=remaining(start, 59, time.monotonic()))
        after = _api("GET", CONTAINERS + "json?all=1", timeout=remaining(start, 60, time.monotonic()))
        _docker_ps(output / "docker-after.txt", remaining(start, 60, time.monotonic()))
        publish(output / "docker-after.json", {"containers": after})
        state["containers_clean"] = (type(after) is list and after == []
                                     and _read(output / "docker-after.txt", output) == b"")
    except (OSError, subprocess.SubprocessError, ValueError):
        state["containers_clean"] = False


def _captured_files(output):
    files = [p for p in output.iterdir() if p.is_file()]
    for role in ("native", "logs", "runtime/generated"):
        files += [p for p in (output / role).rglob("*") if p.is_file()]
    return [artifact(p, output) for p in sorted(files)]


def _save_request_clock(event, clocks, wall, mono, state=None):
    clocks.update({event + '_wall': wall, event + '_monotonic': mono})
    if state is not None:
        state['create_status'] = 'pending'
    return mono


def _request_clock(samples, event, clocks, state=None):
    if samples is None:
        raise ValueError('dram_live_series_required')
    wall, mono = time.time(), time.monotonic()
    evidence.launch_freshness(samples, wall, mono)
    return _save_request_clock(event, clocks, wall, mono, state)


def _device_free_request_clock(event, clocks, state=None):
    return _save_request_clock(event, clocks, time.time(), time.monotonic(), state)


def _create_container(name, payload, start, state, samples, clocks, observed):
    def boundary():
        if observed:
            _request_clock(samples, 'create', clocks, state)
        else:
            _device_free_request_clock('create', clocks, state)
    timeout = remaining(start, 15, time.monotonic())
    if observed:
        return _api('POST', CONTAINERS + 'create?name=' + name, payload,
                    timeout=timeout, before_request=boundary)
    boundary()
    return _api('POST', CONTAINERS + 'create?name=' + name, payload, timeout=timeout)


def _require_capture_observations(state, observations):
    if not state['device_free'] and observations is None:
        raise ValueError('dram_live_series_required')


def capture(phase, identity, assets, output, *, listing=False, approved_payload=None, observations=None):
    identity = _identity(identity)  # Validate BEFORE creating any files or process arguments.
    output = _owned(output)
    assets = _owned(assets)
    if assets != ASSETS.absolute():
        raise ValueError("only receipted assets are allowed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "native").mkdir(mode=0o700)
    payload = approved_payload or request(phase, identity, assets, output, uid=os.getuid(), gid=os.getgid())
    if listing:
        payload["Cmd"] += ["--list-tests"]
    if observations is not None:
        publish(output / "observations-before.json", observations)
    start, name, cid = time.monotonic(), "issue30-" + identity, None
    started_wall, create_clock = time.time(), {}
    binary = assets / BINARIES[phase]
    state = {"schema": "issue30-execution-v1", "attempt": identity, "started": False, "exit": None,
             "timed_out": False, "terminated": False, "create_status": "not_requested",
             "device_free": not payload["HostConfig"]["Devices"],
             "watcher_clean": None, "containers_clean": False, "image": IMAGE,
             "binary_sha256": artifact(binary, assets)["sha256"] if binary.is_file() else None,
             "argv": payload["Cmd"], "environment": payload["Env"]}
    try:
        _require_capture_observations(state, observations)
        _docker_ps(output / "docker-before.txt", remaining(start, 5, time.monotonic()))
        before = _api("GET", CONTAINERS + "json?all=1", timeout=remaining(start, 8, time.monotonic()))
        publish(output / "docker-before.json", {"containers": before})
        if (type(before) is not list or before != []
                or _read(output / "docker-before.txt", output) != b""):
            raise ValueError("preexisting_containers")
        samples = None
        if observations is not None:
            _, inputs = _slots(identity, True)
            samples = _admit_series(phase, inputs, observations)
        created = _create_container(name, payload, start, state, samples, create_clock,
                                    observations is not None)
        cid = _container_id(created["Id"])
        state['create_status'] = 'identified'
        if observations is not None:
            samples = _admit_series(phase, inputs, observations)
            _run_client(name, cid, output, start, state,
                        before_start=lambda: _request_clock(samples, 'launch', create_clock))
        else:
            _device_free_request_clock('launch', create_clock)
            _run_client(name, cid, output, start, state)
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as exc:
        state["error_type"] = type(exc).__name__
        if state['create_status'] == 'pending':
            state['create_status'] = 'unknown'
    finally:
        _cleanup(name, cid, output, start, state)
        if state['create_status'] == 'unknown':
            state['containers_clean'] = False
        finished = time.monotonic()
        state["elapsed_seconds"] = finished - start
        publish(output / "capture-clock.json", {"schema": "issue30-capture-clock-v1",
                "attempt": identity, "start_wall": started_wall, "end_wall": time.time(),
                "start_monotonic": start, "end_monotonic": finished} | create_clock)
        state["artifacts"] = _captured_files(output)
        publish(output / EXECUTION_FILE, state)
    return state


def _load(path, base=None):
    value = formats.load(_read(path, base, 4 * 1024 * 1024))
    if not isinstance(value, dict):
        raise TypeError("evidence must be an object")
    return value


def _slots(identity, board=False):
    identity = _identity(identity)
    output = WORK / ("runs" if board else "g0/checks") / identity
    inputs = WORK / "observations" / identity
    return output, inputs


def _git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def _renewed_gate(phase, inputs):
    if phase == "B":
        gate = _load(ROOT / "docs/measurements/issue196/morning-admission.json", ROOT)
        if gate.get("owner_approved") is not True:
            candidate = _load(inputs / BEFORE_FILE)
            if "owner_unit" not in candidate:
                raise ValueError("B_v2_requires_new_owner_and_fresh_A_decision")
            _renewed_unit(candidate)


def _admit_series(phase, inputs, env):
    if "owner_unit" not in env:
        raise ValueError('dram_live_series_required')
    samples = _raw_series(env, phase, "before", inputs)
    if phase == "B":
        _, ain = _slots(env["owner_unit"]["A"], True)
        a_after = _load(ain / "after.json")
        evidence.ordered(_raw_series(a_after, "A", "after", ain), samples)
    evidence.launch_freshness(samples, time.time(), time.monotonic())
    return samples


def _authorize(phase, identity, inputs, output):
    _renewed_gate(phase, inputs)
    env = _load(inputs / BEFORE_FILE)
    _admit_series(phase, inputs, env)
    topology.validate_input(env)
    _environment(env, identity, phase)
    receipt = _load(CURRENT_RECEIPT, ROOT)
    if _git(ROOT, "status", "--porcelain"):
        raise ValueError("hardware capture requires a clean committed tree")
    if env.get("boot") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        raise ValueError("different boot")
    loaded = Path("/sys/module/tenstorrent/version")
    if not loaded.is_file() or loaded.read_text().strip() != env.get("kmd"):
        raise ValueError("loaded KMD mismatch")
    for source, pin in ((SOURCE_ROOT, SOURCE_COMMIT), (SOURCE_ROOT / "tt_metal/third_party/umd", UMD_COMMIT)):
        if _git(source, "rev-parse", "HEAD") != pin or _git(source, "status", "--porcelain"):
            raise ValueError("pinned source is stale or dirty")
    if receipt.get("source_commit") != SOURCE_COMMIT or receipt.get("runtime_library_sha256") != LIBRARY_SHA256:
        raise ValueError("receipt source/library pin mismatch")
    if phase == "B":
        _transfer_binding(env, _bound_discovery(inputs), b"")
    payload = board_request(phase, identity, ASSETS, output, approval=_load(inputs / "approval.json"),
                            environment=env, receipt=receipt, head=_git(ROOT, "rev-parse", "HEAD"))
    return payload, env


def _observations(inputs, review):
    return [formats.load(_verified(inputs / (role + ".json"), inputs,
                                   {"sha256": review.get(role + "_sha256")}))
            for role in ("before", "after") ]


def _raw_series(env, phase, role, inputs):
    return evidence.series(env, phase, role, lambda name: _read(inputs / name, inputs))


def _capture_clock(output, execution):
    name = 'capture-clock.json'
    if (output / name).exists() or any(e['path'] == name for e in execution['artifacts']):
        return formats.load(_captured(output, execution, name))
    return evidence.ABSENT_CLOCK


def _observed_clock_proof(phase, output, execution, before, after):
    clock = _capture_clock(output, execution)
    if clock is evidence.ABSENT_CLOCK:
        return
    raw = _captured(output, execution, 'native/gtest.xml')
    _xml(phase, raw)
    evidence.capture_window(raw, execution, clock)
    if 'owner_unit' in before:
        evidence.summary_clocks(clock, before, after)


def _series_proof(phase, output, inputs, execution, before, after):
    if "owner_unit" not in before:
        raise ValueError('dram_live_series_required')
    first = _raw_series(before, phase, "before", inputs)
    last = _raw_series(after, phase, "after", inputs)
    clock = _capture_clock(output, execution)
    native_xml = _captured(output, execution, "native/gtest.xml")
    _xml(phase, native_xml)
    evidence.bracket(first, last, native_xml, execution, clock)
    if phase == "B":
        aout, ain = _slots(before["owner_unit"]["A"], True)
        _, areview = _review("A", before["owner_unit"]["A"], aout, ain)
        a_before, a_after = _observations(ain, areview)
        evidence.ordered(_raw_series(a_before, "A", "before", ain),
                         _raw_series(a_after, "A", "after", ain), first, last)


def _review_execution(phase, identity, output, review):
    formats.review(review)
    raw_execution = _verified(output / EXECUTION_FILE, output,
                              {"sha256": review['execution_sha256']})
    execution = formats.load(raw_execution)
    formats.execution(execution)
    if execution.get("schema") != "issue30-execution-v1" or execution.get("attempt") != identity:
        raise ValueError("execution schema/attempt mismatch")
    if (review.get("schema") != "issue30-review-v1" or review.get("attempt") != identity
            or review.get("execution_sha256") != digest(raw_execution)):
        raise ValueError("review execution binding")
    if execution.get("device_free") is not False:
        raise ValueError("device-free capture cannot qualify as board evidence")
    _captured_command(phase, execution)
    receipt = _load(CURRENT_RECEIPT, ROOT)
    if (type(receipt) is not dict or type(receipt.get('binaries')) is not dict
            or execution.get("image") != IMAGE
            or execution.get("binary_sha256") != receipt["binaries"][BINARIES[phase]]):
        raise ValueError("execution_provenance")
    return execution


def _captured_command(phase, execution):
    runtime = ['timeout', '--signal=KILL', '40s', 'python3',
               '/harness/enodia/tt/bench/issue30_runtime.py', phase]
    original = ['timeout', '--signal=KILL', '40s', '/assets/' + BINARIES[phase],
                '--gtest_filter=' + TESTS[phase], '--gtest_output=xml:/capture/native/gtest.xml']
    if not any(formats.equal(execution['argv'], command) for command in (runtime, original)):
        raise ValueError('execution_provenance')
    pairs = [item.split('=', 1) for item in execution['environment']]
    root = dict(pairs)['TT_METAL_RUNTIME_ROOT']
    expected = '/capture/runtime' if execution['argv'] == runtime else '/opt/venv/lib/python3.12/site-packages/ttnn'
    if root != expected:
        raise ValueError('execution_provenance')


def _container_observations(output, execution):
    for role in ("before", "after"):
        listing = formats.load(_captured(output, execution, "docker-" + role + ".json"))
        if not formats.equal(listing, {"containers": []}) or _captured(output, execution, "docker-" + role + ".txt") != b"":
            raise ValueError("containers")


def _review_capture(phase, identity, output, inputs):
    review = _load(inputs / "review.json")
    execution = _review_execution(phase, identity, output, review)
    _container_observations(output, execution)
    for entry in execution['artifacts']:
        if phase != 'B' or entry['path'] != 'logs/generated/watcher/watcher.log':
            _captured(output, execution, entry['path'])
    before, after = _observations(inputs, review)
    formats.environment(before)
    formats.environment(after)
    captured_before = formats.load(_captured(output, execution, "observations-before.json"))
    if not formats.equal(captured_before, before):
        raise ValueError("captured before observation mismatch")
    _observed_clock_proof(phase, output, execution, before, after)
    clean = _watcher_review(phase, output, execution, review)
    return execution | {"watcher_clean": clean}, review


def _review(phase, identity, output, inputs):
    execution, review = _review_capture(phase, identity, output, inputs)
    before, after = _observations(inputs, review)
    _series_proof(phase, output, inputs, execution, before, after)
    return execution, review


def _watcher_review(phase, output, execution, review):
    if phase != "B":
        return review.get("watcher_clean")
    try:
        raw = _captured(output, execution, "logs/generated/watcher/watcher.log")
    except ValueError:
        return False
    scan = diagnostics.watcher_observation(raw, execution.get('elapsed_seconds'))
    return (review.get("watcher_clean") is True
            and review.get("watcher_sha256") == digest(raw) and scan["scan_passed"])


def _judgment(phase, identity, output, inputs, discovery=None):
    """Read-only adjudication shared by live judge and offline evidence replay.

    This does not issue/reuse physical authority: live B separately requires
    _bound_discovery and approval checks before this evidence-only boundary.
    """
    execution, review = _review(phase, identity, output, inputs)
    raw = _native(output, execution)
    before, after = _observations(inputs, review)
    if phase == "B" and "owner_unit" in before:
        _fresh_evidence(before, discovery)
    result = _capture_result(phase, identity, before, after, raw, execution, discovery)
    result["xml_present"] = True
    return result


def _accepted_native_hashes(before, discovery):
    if 'owner_unit' in before:
        return discovery.get('evidence_sha256')
    owner = _load(ROOT / OWNER_ACCEPTANCE, ROOT)
    if not formats.equal(discovery, owner):
        raise ValueError('owner_acceptance_binding')
    original = owner.get('original_verdict')
    if type(original) is not dict:
        raise ValueError('A_native_binding')
    return original.get('evidence_sha256')


def _discovery_uids(before, discovery):
    identity = (before['owner_unit']['A'] if 'owner_unit' in before else discovery['attempt'])
    expected = _accepted_native_hashes(before, discovery)
    output, inputs = _slots(identity, True)
    execution, _ = _review_capture('A', identity, output, inputs)
    raw = _native(output, execution)
    actual = {name: digest(data) for name, data in zip(('xml', 'stdout', 'stderr'), raw, strict=True)}
    if not formats.equal(actual, expected):
        raise ValueError('A_native_binding')
    warnings = []
    lines = _messages(before, raw[1], warnings) + _messages(before, raw[2], warnings)
    headers = [_chip(body) for body in lines if body.startswith('Chip: ')]
    return _checked_uids(before, headers, warnings)


def _capture_result(phase, identity, before, after, raw, execution, discovery):
    uids = _discovery_uids(before, discovery) if phase == 'B' and discovery else None
    return evaluate(phase, identity, before, after, *raw, execution, discovery, uids)


def _judge(phase, identity, output, inputs):
    discovery = _bound_discovery(inputs) if phase == "B" else None
    result = _judgment(phase, identity, output, inputs, discovery)
    publish(output / "result.json", result)
    return 0 if result["reason"] == "accepted" else 2


def _capture_cli(args, output, inputs):
    payload, observation = None, None
    if args.approved_board:
        if args.list_tests:
            raise ValueError("hardware approval excludes listing")
        payload, observation = _authorize(args.phase, args.attempt, inputs, output)
    state = capture(args.phase, args.attempt, ASSETS, output, listing=args.list_tests,
                    approved_payload=payload, observations=observation)
    if not args.list_tests:
        return 2  # Capture alone never admits a hardware transfer.
    try:
        _execution(state, 'A')  # Listing has no application-Watcher requirement.
    except ValueError:
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "capture", "judge"))
    parser.add_argument("--phase", choices=TESTS, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--list-tests", action="store_true")
    parser.add_argument("--approved-board", action="store_true")
    args = parser.parse_args()
    try:
        identity = _identity(args.attempt)
        output, inputs = _slots(identity, args.approved_board or args.action == "judge")
        if args.action == "plan":
            payload = request(args.phase, identity, ASSETS, output, uid=os.getuid(), gid=os.getgid())
            payload["HostConfig"]["Binds"] = ["ASSETS:/assets:ro", "ATTEMPT:/capture:rw",
                                              "HARNESS:/harness:ro", "PINNED_SOURCE:/source:ro"]
            print(json.dumps(payload, indent=2))
            return 0
        if args.action == "capture":
            return _capture_cli(args, output, inputs)
        return _judge(args.phase, identity, output, inputs)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": "issue30-cli-refusal-v1", "error_type": type(exc).__name__}))
        return 2  # No exception text can leak host paths into public summaries.


if __name__ == "__main__":
    raise SystemExit(main())
