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
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree as ET

from enodia.tt.bench import issue30_topology as topology

IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "out/issue30"
ASSETS = WORK / "g0/assets"
SOURCE_ROOT = WORK / "tt-metal"
CURRENT_RECEIPT = ROOT / "docs/measurements/issue30-g0/receipt-v4.json"
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
BEFORE_FILE = "before.json"
MASKS = ["/sys", "/proc/bus", "/proc/driver", "/proc/acpi", "/proc/asound",
         "/proc/interrupts", "/proc/kcore", "/proc/keys", "/proc/latency_stats",
         "/proc/timer_list", "/proc/timer_stats", "/proc/sched_debug", "/proc/scsi"]
PROGRAM_FILES = ("enodia/tt/bench/issue30.py", "enodia/tt/bench/issue30_build/CMakeLists.txt",
                 "tools/build_issue30.py", "tests/test_issue30.py", "tests/test_issue30_capture.py",
                 "pyproject.toml", "uv.lock", "enodia/tt/bench/issue30_topology.py",
                 "tests/test_issue30_topology.py", topology.POLICY_FILE)


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


def publish(path: Path, record: dict) -> None:
    """Linux no-replace atomic files; no symlink/overwrite fallback."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(path.name)
    if path.absolute() != path.resolve():
        raise ValueError("redirected publication")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as output:
        output.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temporary), -100, os.fsencode(path), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remaining(start: float, limit: int, now: float) -> float:
    return max(0.0, start + limit - now)


def request(phase: str, identity: str, assets: Path, capture: Path, *, uid: int, gid: int) -> dict:
    _identity(identity)
    if uid <= 0 or gid < 0:
        raise ValueError("non-root identity required")
    scratch = f"rw,nosuid,nodev,noexec,size=256m,mode=0700,uid={uid},gid={gid}"
    return {"Image": IMAGE, "User": f"{uid}:{gid}", "WorkingDir": "/capture",
            "Env": ["TT_METAL_WATCHER=1", "HOME=/scratch", "LD_BIND_NOW=1", "TMPDIR=/scratch",
                    "TT_METAL_RUNTIME_ROOT=/opt/venv/lib/python3.12/site-packages/ttnn",
                    "TT_METAL_CACHE=/capture/cache", "TT_METAL_LOGS_PATH=/capture/logs"],
            "Cmd": ["timeout", "--signal=KILL", "40s", f"/assets/{BINARIES[phase]}",
                    f"--gtest_filter={TESTS[phase]}", "--gtest_output=xml:/capture/native/gtest.xml"],
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Devices": [],
                           "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                           "MaskedPaths": MASKS, "Tmpfs": {"/scratch": scratch},
                           "Binds": [f"{assets.resolve()}:/assets:ro", f"{capture.resolve()}:/capture:rw"]}}


def _approval(phase, identity, approval, env, receipt, head):
    expected = {"schema": "issue30-owner-approval-v1", "attempt": identity, "phase": phase,
                "head": head, "boot": env["boot"],
                "environment_sha256": digest(json.dumps(env, sort_keys=True).encode()),
                "receipt_sha256": digest(json.dumps(receipt, sort_keys=True).encode())}
    if approval != expected:
        raise ValueError("approval_binding")
    if receipt.get("schema") != "issue30-g0-v1" or receipt.get("all_passed") is not True:
        raise ValueError("g0_not_ready")
    if receipt.get("current_wrapper_sources") != program_sources():
        raise ValueError("receipt_source_binding")


def _binary_binding(assets, receipt):
    for name in BINARIES.values():
        path = assets / name
        if path.is_symlink() or digest(path.read_bytes()) != receipt.get("binaries", {}).get(name):
            raise ValueError("binary_binding")


def board_request(phase: str, identity: str, assets: Path, output: Path, *,
                  approval: dict, environment: dict, receipt: dict, head: str) -> dict:
    _environment(environment, identity)
    _approval(phase, identity, approval, environment, receipt, head)
    _binary_binding(assets, receipt)
    payload = request(phase, identity, assets, output, uid=os.getuid(), gid=os.getgid())
    nodes = [c.get("node", "") for c in environment["cards"]]
    if len(set(nodes)) != 2 or any(not re.fullmatch(r"/dev/tenstorrent/\d{1,6}", n, re.ASCII) for n in nodes):
        raise ValueError("device_mapping")
    host = payload["HostConfig"]
    host["Devices"] = [{"PathOnHost": n, "PathInContainer": n, "CgroupPermissions": "rw"} for n in nodes]
    host["MaskedPaths"] = [p for p in MASKS if p not in ("/sys", "/proc/bus", "/proc/driver")]
    host["Binds"] += [f"{SOURCE_ROOT}:/source:ro", "/dev/hugepages-1G:/dev/hugepages-1G:rw"]
    host["Ulimits"] = [{"Name": "memlock", "Soft": -1, "Hard": -1}]
    payload["Env"] += ["TT_VISIBLE_DEVICES=" + ",".join(c[1] for c in CARDS), "TT_METAL_KERNEL_PATH=/source"]
    return payload


def _bundle_report_name(value):
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}(?:\.0)?", value, re.ASCII):
        raise ValueError("environment")
    return ".".join(str(int(x)) for x in value.split(".")[:3])


def _card(card):
    erisc, cmfw = card.get("erisc", ""), card.get("cmfw", "")
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}", erisc, re.ASCII):
        raise ValueError("environment")
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}(?:\.\d{1,3})?", cmfw, re.ASCII):
        raise ValueError("environment")
    if not any(map(int, cmfw.split("."))) or tuple(map(int, erisc.split("."))) < (1, 4, 1):
        raise ValueError("environment")
    if tuple(map(int, _bundle_report_name(card.get("bundle", "")).split("."))) < (18, 5, 0):
        raise ValueError("environment")
    if card.get("dram_healthy") is not True or card.get("eth_heartbeat_healthy") is not True:
        raise ValueError("environment")
    if type(card.get("aiclk_mhz")) is not int or card["aiclk_mhz"] <= 0 or type(card.get("pcie_index")) is not int:
        raise ValueError("environment")


def _cards(cards):
    if not isinstance(cards, list) or len(cards) != 2:
        raise ValueError("environment")
    if {c.get("chip") for c in cards} != {0, 1}:
        raise ValueError("environment")
    if {(c.get("serial"), c.get("pci")) for c in cards} != set(CARDS):
        raise ValueError("environment")
    for c in cards:
        _card(c)
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


def _environment(env, identity):
    if (env.get("schema") != "issue30-environment-v1" or env.get("attempt") != identity
            or not env.get("boot") or env.get("image") != IMAGE or env.get("watcher") != "1"):
        raise ValueError("environment")
    kmd = env.get("kmd", "")
    if not re.fullmatch(r"\d{1,3}\.\d{1,3}\.\d{1,3}", kmd, re.ASCII) or tuple(map(int, kmd.split("."))) < (2, 8, 0):
        raise ValueError("environment")
    _cards(env.get("cards"))
    _links(env)


def _xml(phase, raw):
    if len(raw) > 1024 * 1024 or b"<!" in raw:
        raise ValueError("test_result")
    try:
        root = ET.fromstring(raw)
    except ParseError as exc:
        raise ValueError("test_result") from exc
    cases = list(root.iter("testcase"))
    suite, name = TESTS[phase].split(".")
    if (root.tag != "testsuites" or len(list(root.iter("testsuite"))) != 1 or len(cases) != 1):
        raise ValueError("test_result")
    if (cases[0].get("name") != name or cases[0].get("classname") != suite
            or cases[0].get("status") != "run" or cases[0].get("result") != "completed"):
        raise ValueError("test_result")
    if (any(list(root.iter(tag)) for tag in ("failure", "error", "skipped"))
            or any(n.get(k, "0") != "0" for n in root.iter() for k in ("failures", "errors", "disabled", "skipped"))):
        raise ValueError("test_result")


def _number(text):
    if not re.fullmatch(r"\d{1,10}", text, re.ASCII):
        raise ValueError("peer_links")
    return int(text)


def _strip_source(text):
    body, sep, tail = text.rpartition(" (")
    filename, colon, line = tail.removesuffix(")").rpartition(":")
    if sep and tail.endswith(")") and colon and line.isdecimal() and len(filename) <= 256:
        return body
    return text


def _message(line):
    parts = line.split("|", 3)
    if len(parts) == 4:
        return parts[1].strip().casefold(), _strip_source(parts[3].strip())
    if len(parts) == 2:
        return parts[0].strip().casefold(), _strip_source(parts[1].strip())
    return "", line.strip()


def _chip(text):
    t = text.split()
    if len(t) < 7 or (t[0], t[2], t[4], t[5]) != ("Chip:", "PCIe:", "Unique", "ID:"):
        raise ValueError("chip_identity")
    if not re.fullmatch(r"[0-9a-f]{1,16}", t[6]):
        raise ValueError("chip_identity")
    return _number(t[1]), _number(t[3])


def _link(text):
    t = text.split()
    if len(t) < 8 or (t[0], t[1], t[3], t[5]) != ("eth", "channel", "core", "link"):
        raise ValueError("peer_links")
    if not re.fullmatch(r"\d{1,3}-\d{1,3}", t[4], re.ASCII):
        raise ValueError("peer_links")
    data = {"channel": _number(t[2]), "core": t[4], "state": t[6]}
    if t[6:] == [DOWN, "(QSFP)"]:
        return data
    if len(t) < 14 or t[6:10] != ["UP", "(QSFP),", "connected", "to"] or t[-2] != "core":
        raise ValueError("peer_links")
    if not re.fullmatch(r"\d{1,3}-\d{1,3}", t[-1], re.ASCII):
        raise ValueError("peer_links")
    if t[10] == "Chip":
        return data | {"peer": _number(t[11]), "peer_core": t[-1]}
    if t[10:12] == ["Unique", "ID:"] and re.fullmatch(r"[a-f0-9]{1,16}", t[12]):
        return data | {"peer_unique_id": t[12], "peer_core": t[-1]}
    raise ValueError("peer_links")


def _warning(env, body):
    header, sep, link = body.partition(" eth channel ")
    try:
        chip, pcie = _chip(header)
        parsed = _link("eth channel " + link)
    except ValueError as exc:
        raise ValueError("diagnostic") from exc
    indices = {c["chip"]: c["pcie_index"] for c in env["cards"]}
    allowed = ([chip, parsed["channel"]] in env.get("uncabled", [])
               or "topology" in env and parsed["channel"] in topology.CHANNEL_PORT)
    if not sep or parsed["state"] != DOWN or indices.get(chip) != pcie or not allowed:
        raise ValueError("diagnostic")
    return {"chip": chip} | parsed


def _messages(env, raw, warnings=None):
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("diagnostic")
    try:
        text = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8"))
    except UnicodeError as exc:
        raise ValueError("diagnostic") from exc
    messages = []
    for line in text.splitlines():
        level, body = _message(line)
        if re.search(r"\b(?:error|fatal|critical)\b", line, re.IGNORECASE):
            raise ValueError("diagnostic")
        if level == "warning":
            warning = _warning(env, body)
            if warnings is not None:
                warnings.append(warning)
        else:
            messages.append(body)
    return messages


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
            found.append({k: v for k, v in data.items() if k not in ("state", "retrain_count")})
    canonical = lambda xs: sorted(json.dumps(x, sort_keys=True) for x in xs)
    if canonical(found) != canonical(env["links"]):
        raise ValueError("peer_links")


def _health_records(env, lines, observed, warnings):
    headers, chip = [], None
    for index, body in enumerate(lines):
        if body.startswith("Chip: "):
            chip, pcie = _chip(body)
            headers.append((chip, pcie))
        elif body.startswith("eth channel "):
            observed.append(_observed_link(lines, index, chip))
    expected = {(c["chip"], c["pcie_index"]) for c in env["cards"]}
    if len(headers) != 2 or set(headers) != expected:
        raise ValueError("chip_identity")
    if "topology" in env:
        topology.check_warnings(observed, warnings)
        return topology.classify(observed)
    _legacy_links(env, observed)
    return None


def _health(env, raw, observed):
    warnings = []
    lines = _messages(env, raw, warnings)
    if [x for x in lines if x.startswith("Found ")] != ["Found 2 chips in cluster:"]:
        raise ValueError("chip_count")
    prefix = "Established firmware bundle version: "
    bundles = [x.removeprefix(prefix) for x in lines if x.startswith(prefix)]
    if bundles != [_bundle_report_name(env["cards"][0]["bundle"])]:
        raise ValueError("bundle")
    return _health_records(env, lines, observed, warnings)


def baseline_digest(env):
    return digest(json.dumps({k: v for k, v in env.items() if k != "attempt"}, sort_keys=True).encode())


def _execution(execution):
    if (type(execution.get("exit")) is not int or execution["exit"] != 0
            or execution.get("timed_out") is not False or execution.get("terminated") is not True
            or execution.get("started") is not True or execution.get("client_exit") != 0):
        raise ValueError("execution")
    elapsed = execution.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not 0 <= elapsed <= 60:
        raise ValueError("execution")
    if execution.get("watcher_clean") is not True:
        raise ValueError("watcher")
    if execution.get("containers_clean") is not True:
        raise ValueError("containers")


def _post(env, after, identity):
    _environment(env, identity)
    _environment(after, identity)
    if env != after:
        raise ValueError("environment")


def _transfer_binding(before, discovery, raw):
    if (not discovery or discovery.get("phase") != "A" or discovery.get("verdict") != ACCEPTED["A"]
            or discovery.get("baseline_sha256") != baseline_digest(before)):
        raise ValueError("discovery_binding")
    if "topology" in before:
        topology.require_accepted_discovery(discovery)
    if re.search(rb"\b(?:warning|error|fatal|critical)\b", raw, re.IGNORECASE):
        raise ValueError("diagnostic")


def evaluate(phase, identity, before, after, xml, stdout, stderr, execution, discovery=None):
    reason, decision, observed = "accepted", None, []
    try:
        _execution(execution)
        _xml(phase, xml)
        _post(before, after, identity)
        if phase == "A":
            decision = _health(before, stdout + b"\n" + stderr, observed)
            if decision is not None:
                reason = decision["reason"]
        else:
            _transfer_binding(before, discovery, stdout + stderr)
    except (ValueError, KeyError, TypeError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "environment"
    result = {"schema": "issue30-verdict-v1", "attempt": identity, "phase": phase,
              "verdict": ACCEPTED[phase] if reason == "accepted" else "rejected", "reason": reason,
              "scope": SCOPES[phase], "baseline_sha256": baseline_digest(before),
              "evidence_sha256": {"xml": digest(xml), "stdout": digest(stdout), "stderr": digest(stderr)}}
    if "topology" in before:
        result |= {"schema": "issue30-verdict-v2", "topology": decision, "observed_qsfp": observed}
        if decision is not None and decision["reason"] == reason:
            result["verdict"] = {"accepted": ACCEPTED[phase], "not_established": "link_not_established",
                                 "unexpected": "unexpected_topology"}.get(decision["outcome"], "rejected")
    return result


def _api(method, path, body=None, timeout=5):
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker API budget", timeout)
    command = ["curl", "-fsS", "--noproxy", "*", "--max-time", str(timeout), "--unix-socket",
               "/var/run/docker.sock", "-X", method, "-H", "Content-Type: application/json"]
    if body is not None:
        command += ["--data-binary", "@-"]
    command += ["http://localhost/v1.52/" + path]
    done = subprocess.run(command, input=json.dumps(body).encode() if body is not None else None,
                          capture_output=True, timeout=timeout + 0.1, check=True)
    return json.loads(done.stdout) if done.stdout else {}


def _docker_ps(path, timeout):
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker ps budget", timeout)
    done = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.State}}\t{{.Image}}"],
                          capture_output=True, check=True, timeout=timeout)
    with path.open("xb") as stream:
        stream.write(done.stdout)


def _container_id(value):
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid daemon container identity")
    return value


def _run_client(name, cid, output, start, state):
    with (output / "stdout.bin").open("xb") as stdout, (output / "stderr.bin").open("xb") as stderr:
        client = subprocess.Popen(["docker", "start", "-a", "--", name], stdout=stdout, stderr=stderr)
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


def _cleanup(name, cid, output, start, state):
    try:
        if cid is not None:
            _api("DELETE", CONTAINERS + cid + "?force=1", timeout=remaining(start, 59, time.monotonic()))
        after = _api("GET", CONTAINERS + "json?all=1", timeout=remaining(start, 60, time.monotonic()))
        _docker_ps(output / "docker-after.txt", remaining(start, 60, time.monotonic()))
        publish(output / "docker-after.json", {"containers": after})
        state["containers_clean"] = not any("/" + name in c.get("Names", []) for c in after)
    except (OSError, subprocess.SubprocessError, ValueError):
        state["containers_clean"] = False


def _captured_files(output):
    files = [p for p in output.iterdir() if p.is_file()]
    for role in ("native", "logs"):
        files += [p for p in (output / role).rglob("*") if p.is_file()]
    return [artifact(p, output) for p in sorted(files)]


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
        payload["Cmd"] = payload["Cmd"][:4] + ["--gtest_list_tests"]
    if observations is not None:
        publish(output / "observations-before.json", observations)
    start, name, cid = time.monotonic(), "issue30-" + identity, None
    binary = assets / BINARIES[phase]
    state = {"schema": "issue30-execution-v1", "attempt": identity, "started": False, "exit": None,
             "timed_out": False, "terminated": False, "device_free": not payload["HostConfig"]["Devices"],
             "watcher_clean": None, "containers_clean": False, "image": IMAGE,
             "binary_sha256": artifact(binary, assets)["sha256"] if binary.is_file() else None,
             "argv": payload["Cmd"], "environment": payload["Env"]}
    try:
        _docker_ps(output / "docker-before.txt", remaining(start, 5, time.monotonic()))
        before = _api("GET", CONTAINERS + "json?all=1", timeout=remaining(start, 8, time.monotonic()))
        publish(output / "docker-before.json", {"containers": before})
        if before:
            raise ValueError("preexisting_containers")
        created = _api("POST", CONTAINERS + "create?name=" + name, payload, timeout=remaining(start, 15, time.monotonic()))
        cid = _container_id(created["Id"])
        _run_client(name, cid, output, start, state)
    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
        state["error_type"] = type(exc).__name__
    finally:
        _cleanup(name, cid, output, start, state)
        state["elapsed_seconds"] = time.monotonic() - start
        state["artifacts"] = _captured_files(output)
        publish(output / EXECUTION_FILE, state)
    return state


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate evidence key")
        result[key] = value
    return result


def _load(path, base=None):
    value = json.loads(_read(path, base, 4 * 1024 * 1024), object_pairs_hook=_unique_json)
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


def _authorize(phase, identity, inputs, output):
    env = _load(inputs / BEFORE_FILE)
    topology.validate_input(env)
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
        _transfer_binding(env, _load(inputs / "discovery.json"), b"")
    payload = board_request(phase, identity, ASSETS, output, approval=_load(inputs / "approval.json"),
                            environment=env, receipt=receipt, head=_git(ROOT, "rev-parse", "HEAD"))
    return payload, env


def _review(phase, identity, output, inputs):
    execution = _load(output / EXECUTION_FILE)
    review = _load(inputs / "review.json")
    if execution.get("schema") != "issue30-execution-v1" or execution.get("attempt") != identity:
        raise ValueError("execution schema/attempt mismatch")
    if (review.get("schema") != "issue30-review-v1" or review.get("attempt") != identity
            or review.get("execution_sha256") != digest(_read(output / EXECUTION_FILE))):
        raise ValueError("review execution binding")
    if execution.get("device_free") is not False:
        raise ValueError("device-free capture cannot qualify as board evidence")
    if _load(output / "observations-before.json") != _load(inputs / BEFORE_FILE):
        raise ValueError("captured before observation mismatch")
    if phase == "B":
        watcher = output / "logs/generated/watcher/watcher.log"
        if not _read(watcher) or review.get("watcher_sha256") != artifact(watcher, output)["sha256"]:
            raise ValueError("B requires a captured reviewed Watcher log")
    return execution | {"watcher_clean": review.get("watcher_clean")}, review


def _judge(phase, identity, output, inputs):
    execution, review = _review(phase, identity, output, inputs)
    for field in ("before", "after"):
        if review.get(field + "_sha256") != digest(_read(inputs / (field + ".json"))):
            raise ValueError("review observation fingerprint mismatch")
    discovery = _load(inputs / "discovery.json") if phase == "B" else None
    result = evaluate(phase, identity, _load(inputs / BEFORE_FILE), _load(inputs / "after.json"),
                      _read(output / "native/gtest.xml"), _read(output / "stdout.bin"),
                      _read(output / "stderr.bin"), execution, discovery)
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
    success = (args.list_tests and state["exit"] == 0 and state["terminated"]
               and not state["timed_out"] and state["containers_clean"] and state["elapsed_seconds"] <= 60)
    return 0 if success else 2  # Capture alone never admits a hardware transfer.


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
            payload["HostConfig"]["Binds"] = ["ASSETS:/assets:ro", "ATTEMPT:/capture:rw"]
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
