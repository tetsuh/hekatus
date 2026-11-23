"""Issue #30 attempt-bound capture and fail-closed source-report validation.

Import/planning never opens hardware. Default capture is device-free only.
Hardware capture requires a separate attempt/head/boot/environment/artifact-bound
owner approval and a successful G0 receipt; this preparation grants neither.
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

IMAGE = "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621"
CARDS = (("000004033191411c", "0000:06:00.0"), ("0000040331914124", "0000:09:00.0"))
TESTS = {"A": "Cluster.ReportSystemHealth", "B": "Fabric1DFixture.TestUnicastConnAPI1DSmoke"}
BINARIES = {"A": "test_system_health", "B": "fabric_smoke_tests"}
SOURCE_COMMIT = "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd"
UMD_COMMIT = "bc77ac2f7382ffeb9fa5430055f0854d53979489"
LIBRARY_SHA256 = "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141"
SOURCE_ROOT = Path(__file__).resolve().parents[3] / "out/issue30/tt-metal"
DEADLINES = {"binary": 40, "container": 50, "cleanup": 60}
GOOD_EXECUTION = {"exit": 0, "client_exit": 0, "started": True, "elapsed_seconds": 1.0,
                  "timed_out": False, "terminated": True,
                  "watcher_clean": True, "containers_clean": True}
MASKS = ["/sys", "/proc/bus", "/proc/driver", "/proc/acpi", "/proc/asound",
         "/proc/interrupts", "/proc/kcore", "/proc/keys", "/proc/latency_stats",
         "/proc/timer_list", "/proc/timer_stats", "/proc/sched_debug", "/proc/scsi"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(path: Path, root: Path) -> dict:
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact escapes attempt")
    hashed = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            hashed.update(block)
    return {"path": path.relative_to(root).as_posix(), "bytes": count, "sha256": hashed.hexdigest()}


def publish(path: Path, record: dict) -> None:
    """Linux no-replace atomic publication, following ADR-0018's transport rule."""
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path.name)
    with temporary.open("xb") as output:
        output.write((json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
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
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", identity):
        raise ValueError("unsafe attempt identity")
    if uid <= 0 or gid < 0:
        raise ValueError("non-root identity required")
    return {"Image": IMAGE, "User": f"{uid}:{gid}", "WorkingDir": "/capture",
            "Env": ["TT_METAL_WATCHER=1", "HOME=/tmp", "LD_BIND_NOW=1",
                    "TT_METAL_RUNTIME_ROOT=/opt/venv/lib/python3.12/site-packages/ttnn",
                    "TT_METAL_CACHE=/capture/cache", "TT_METAL_LOGS_PATH=/capture/logs"],
            "Cmd": ["timeout", "--signal=KILL", "40s", f"/assets/{BINARIES[phase]}",
                    f"--gtest_filter={TESTS[phase]}", "--gtest_output=xml:/capture/native/gtest.xml"],
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Devices": [],
                           "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                           "MaskedPaths": MASKS, "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=256m"},
                           "Binds": [f"{assets.resolve()}:/assets:ro", f"{capture.resolve()}:/capture:rw"]}}


def board_request(phase: str, identity: str, assets: Path, output: Path, *,
                  approval: dict, environment: dict, receipt: dict, head: str) -> dict:
    _environment(environment, identity)
    expected = {"schema": "issue30-owner-approval-v1", "attempt": identity, "phase": phase,
                "head": head, "boot": environment["boot"],
                "environment_sha256": digest(json.dumps(environment, sort_keys=True).encode()),
                "receipt_sha256": digest(json.dumps(receipt, sort_keys=True).encode())}
    if approval != expected:
        raise ValueError("approval_binding")
    if receipt.get("schema") != "issue30-g0-v1" or receipt.get("all_passed") is not True:
        raise ValueError("g0_not_ready")
    for name in BINARIES.values():
        path = assets / name
        if path.is_symlink() or digest(path.read_bytes()) != receipt.get("binaries", {}).get(name):
            raise ValueError("binary_binding")
    payload = request(phase, identity, assets, output, uid=os.getuid(), gid=os.getgid())
    nodes = [c.get("node", "") for c in environment["cards"]]
    if len(set(nodes)) != 2 or any(not re.fullmatch(r"/dev/tenstorrent/\d+", n) for n in nodes):
        raise ValueError("device_mapping")
    host = payload["HostConfig"]
    host["Devices"] = [{"PathOnHost": n, "PathInContainer": n, "CgroupPermissions": "rw"} for n in nodes]
    host["MaskedPaths"] = [p for p in MASKS if p not in ("/sys", "/proc/bus", "/proc/driver")]
    # This source role is fixed by the repository, never inherited from TT env.
    host["Binds"] += [f"{SOURCE_ROOT}:/source:ro",
                      "/dev/hugepages-1G:/dev/hugepages-1G:rw"]
    host["Ulimits"] = [{"Name": "memlock", "Soft": -1, "Hard": -1}]
    payload["Env"] += ["TT_VISIBLE_DEVICES=" + ",".join(c[1] for c in CARDS),
                       "TT_METAL_KERNEL_PATH=/source"]
    return payload


def _bundle_report_name(value: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.0)?", value):
        raise ValueError("environment")
    return ".".join(str(int(x)) for x in value.split(".")[:3])


def _environment(env: dict, identity: str) -> None:
    if (env.get("schema") != "issue30-environment-v1" or env.get("attempt") != identity
            or not env.get("boot") or env.get("image") != IMAGE or env.get("watcher") != "1"
            or not re.fullmatch(r"\d+\.\d+\.\d+", env.get("kmd", ""))):
        raise ValueError("environment")
    if tuple(map(int, env["kmd"].split("."))) < (2, 8, 0):
        raise ValueError("environment")
    cards = env.get("cards")
    if not isinstance(cards, list) or len(cards) != 2:
        raise ValueError("environment")
    if {c.get("chip") for c in cards} != {0, 1}:
        raise ValueError("environment")
    if {(c.get("serial"), c.get("pci")) for c in cards} != set(CARDS):
        raise ValueError("environment")
    for c in cards:
        if not re.fullmatch(r"\d+\.\d+\.\d+", c.get("erisc", "")):
            raise ValueError("environment")
        version = tuple(int(x) for x in c["erisc"].split("."))
        if tuple(map(int, _bundle_report_name(c.get("bundle", "")).split("."))) < (18, 5, 0):
            raise ValueError("environment")
        cmfw = c.get("cmfw", "")
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.\d+)?", cmfw) or not any(map(int, cmfw.split("."))):
            raise ValueError("environment")
        if (version < (1, 4, 1) or c.get("dram_healthy") is not True
                or c.get("eth_heartbeat_healthy") is not True
                or type(c.get("aiclk_mhz")) is not int or c["aiclk_mhz"] <= 0
                or not c.get("bundle") or not c.get("cmfw") or type(c.get("pcie_index")) is not int):
            raise ValueError("environment")
    if len({c["pcie_index"] for c in cards}) != 2 or len({c["bundle"] for c in cards}) != 1:
        raise ValueError("environment")
    links = env.get("links", [])
    if not links or len({(x["chip"], x["channel"]) for x in links}) != len(links):
        raise ValueError("environment")
    for link in links:
        if (link["chip"] not in (0, 1) or link["peer"] != 1 - link["chip"]
                or not any(x["chip"] == link["peer"] and x["core"] == link["peer_core"]
                           and x["peer_core"] == link["core"] for x in links)):
            raise ValueError("environment")
    if {(x["chip"], x["channel"]) for x in links} & {tuple(x) for x in env.get("uncabled", [])}:
        raise ValueError("environment")


def _xml(phase: str, raw: bytes) -> None:
    if len(raw) > 1024 * 1024 or b"<!" in raw:
        raise ValueError("test_result")
    try:
        root = ET.fromstring(raw)
    except ParseError as exc:
        raise ValueError("test_result") from exc
    if root.tag != "testsuites" or len(list(root.iter("testsuite"))) != 1:
        raise ValueError("test_result")
    cases = list(root.iter("testcase"))
    suite, name = TESTS[phase].split(".")
    if (len(cases) != 1 or cases[0].get("name") != name or cases[0].get("classname") != suite
            or cases[0].get("status") != "run" or cases[0].get("result") != "completed"
            or any(list(root.iter(tag)) for tag in ("failure", "error", "skipped"))
            or any(node.get(k, "0") != "0" for node in root.iter()
                   for k in ("failures", "errors", "disabled", "skipped"))):
        raise ValueError("test_result")


def _health(env: dict, raw: bytes) -> None:
    text = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8", errors="strict"))
    # The only warning exception is an explicitly predeclared uncabled QSFP channel.
    allowed = {tuple(x) for x in env.get("uncabled", [])}
    indices = {c["chip"]: c["pcie_index"] for c in env["cards"]}
    for line in text.splitlines():
        if re.search(r"\b(?:error|fatal|critical)\b", line, re.IGNORECASE):
            raise ValueError("diagnostic")
        if re.search(r"\bwarning\b", line, re.IGNORECASE):
            match = re.search(r"\bwarning\b.*?Chip: (\d+) PCIe: (\d+) Unique ID: [0-9a-f]+[^\n]*?eth channel (\d+) core \d+-\d+ link DOWN/unconnected \(QSFP\)$", line)
            if not match:
                raise ValueError("diagnostic")
            chip, pcie, channel = map(int, match.groups())
            if (chip, channel) not in allowed or indices.get(chip) != pcie:
                raise ValueError("diagnostic")
    # Ignore warning copies, which are not a second source of topology observations.
    report = "\n".join(x for x in text.splitlines() if not re.search(r"\bwarning\b", x, re.IGNORECASE))
    counts = re.findall(r"Found (\d+) chips in cluster:", report)
    if counts != ["2"]:
        raise ValueError("chip_count")
    bundles = re.findall(r"Established firmware bundle version: ([0-9.]+)", report)
    if bundles != [_bundle_report_name(env["cards"][0]["bundle"])]:
        raise ValueError("bundle")
    headers = list(re.finditer(r"^Chip: (\d+) PCIe: (\d+) Unique ID: [0-9a-f]+[^\n]*$", report, re.MULTILINE))
    actual = {(int(x[1]), int(x[2])) for x in headers}
    if len(headers) != 2 or actual != {(c["chip"], c["pcie_index"]) for c in env["cards"]}:
        raise ValueError("chip_identity")
    found = []
    for index, header in enumerate(headers):
        section = report[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(report)]
        if any(x != "0" for x in re.findall(r"Retrain count: (\d+)", section)):
            raise ValueError("retrain")
        pattern = r"eth channel (\d+) core (\d+-\d+) link UP \(QSFP\), connected to Chip (\d+)[^\n]*? core (\d+-\d+)\n\s*Retrain count: 0"
        for match in re.finditer(pattern, section):
            channel, core, peer, peer_core = match.groups()
            found.append({"chip": int(header[1]), "channel": int(channel), "core": core,
                          "peer": int(peer), "peer_core": peer_core})
        for down in re.findall(r"eth channel (\d+).*link DOWN/unconnected", section):
            if (int(header[1]), int(down)) not in allowed:
                raise ValueError("peer_links")
    canonical = lambda xs: sorted(json.dumps(x, sort_keys=True) for x in xs)
    if canonical(found) != canonical(env["links"]):
        raise ValueError("peer_links")


def baseline_digest(env: dict) -> str:
    return digest(json.dumps({k: v for k, v in env.items() if k != "attempt"}, sort_keys=True).encode())


def evaluate(phase: str, identity: str, before: dict, after: dict, xml: bytes,
             stdout: bytes, stderr: bytes, execution: dict, discovery: dict | None = None) -> dict:
    """A positive verdict requires evidence at every independent gate.

    Environment/Watcher observations are supplied files, not generated claims of
    live telemetry. The separate owner-approved capture must produce them.
    """
    reason = "accepted"
    try:
        if (type(execution.get("exit")) is not int or execution["exit"] != 0
                or execution.get("timed_out") is not False or execution.get("terminated") is not True
                or execution.get("started") is not True or execution.get("client_exit") != 0
                or type(execution.get("elapsed_seconds")) not in (int, float)
                or not 0 <= execution["elapsed_seconds"] <= 60):
            raise ValueError("execution")
        _xml(phase, xml)
        _environment(before, identity)
        _environment(after, identity)
        if before != after:
            raise ValueError("environment")
        if execution.get("watcher_clean") is not True:
            raise ValueError("watcher")
        if execution.get("containers_clean") is not True:
            raise ValueError("containers")
        if phase == "A":
            _health(before, stdout + b"\n" + stderr)
        else:
            if (not discovery or discovery.get("phase") != "A"
                    or discovery.get("verdict") != "accepted_discovery"
                    or discovery.get("baseline_sha256") != baseline_digest(before)):
                raise ValueError("discovery_binding")
            if re.search(rb"\b(?:warning|error|fatal|critical)\b", stdout + stderr, re.IGNORECASE):
                raise ValueError("diagnostic")
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "environment"
    return {"schema": "issue30-verdict-v1", "attempt": identity, "phase": phase,
            "verdict": ("accepted_discovery" if phase == "A" else "accepted_transfer")
            if reason == "accepted" else "rejected", "reason": reason,
            "scope": "one discovery" if phase == "A" else "five checked packets; no timing claim",
            "baseline_sha256": baseline_digest(before),
            "evidence_sha256": {"xml": digest(xml), "stdout": digest(stdout), "stderr": digest(stderr)}}


def _api(method: str, path: str, body: dict | None = None, timeout: float = 5) -> dict:
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker API budget", timeout)
    command = ["curl", "-fsS", "--noproxy", "*", "--max-time", str(timeout), "--unix-socket",
               "/var/run/docker.sock", "-X", method, "-H", "Content-Type: application/json"]
    if body is not None:
        command += ["--data-binary", "@-"]
    command += ["http://localhost/v1.52/" + path]
    done = subprocess.run(command, input=json.dumps(body).encode() if body is not None else None,
                          capture_output=True, timeout=max(0.1, timeout + 0.1), check=True)
    return json.loads(done.stdout) if done.stdout else {}


def _docker_ps(path: Path, timeout: float) -> None:
    if timeout <= 0:
        raise subprocess.TimeoutExpired("docker ps budget", timeout)
    result = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.State}}\t{{.Image}}"],
                            capture_output=True, check=True, timeout=timeout)
    with path.open("xb") as stream:
        stream.write(result.stdout)


def capture(phase: str, identity: str, assets: Path, output: Path, *, listing: bool = False,
            approved_payload: dict | None = None, observations: dict | None = None) -> dict:
    """Default device-free; approved payload only comes from board_request."""
    if output.absolute() != output.resolve():
        raise ValueError("redirected attempt directory")
    output.mkdir(exist_ok=False)
    (output / "native").mkdir()
    payload = approved_payload or request(phase, identity, assets, output, uid=os.getuid(), gid=os.getgid())
    if listing:
        payload["Cmd"] = payload["Cmd"][:4] + ["--gtest_list_tests"]
    if observations is not None:
        publish(output / "observations-before.json", observations)
    start = time.monotonic()
    name = "issue30-" + identity
    state = {"schema": "issue30-execution-v1", "attempt": identity, "started": False,
             "exit": None, "timed_out": False, "terminated": False,
             "device_free": not payload["HostConfig"]["Devices"],
             "watcher_clean": None, "containers_clean": False,
             "image": IMAGE, "binary_sha256": artifact(assets / BINARIES[phase], assets)["sha256"]
             if (assets / BINARIES[phase]).is_file() else None,
             "argv": payload["Cmd"], "environment": payload["Env"]}
    created = False
    container_id = name
    try:
        _docker_ps(output / "docker-before.txt", remaining(start, 5, time.monotonic()))
        before = _api("GET", "containers/json?all=1", timeout=remaining(start, 8, time.monotonic()))
        publish(output / "docker-before.json", {"containers": before})
        if before:
            raise ValueError("preexisting_containers")
        creation = _api("POST", "containers/create?name=" + name, payload,
                        timeout=remaining(start, 15, time.monotonic()))
        container_id = creation["Id"]
        created = True
        with (output / "stdout.bin").open("xb") as stdout, (output / "stderr.bin").open("xb") as stderr:
            client = subprocess.Popen(["docker", "start", "-a", name], stdout=stdout, stderr=stderr)
            state["started"] = True
            try:
                state["client_exit"] = client.wait(timeout=remaining(start, 50, time.monotonic()))
            except subprocess.TimeoutExpired:
                state["timed_out"] = True
                try:
                    _api("POST", "containers/" + container_id + "/kill", timeout=remaining(start, 55, time.monotonic()))
                finally:
                    client.kill()
                    client.wait(timeout=remaining(start, 57, time.monotonic()))
        observed = _api("GET", "containers/" + container_id + "/json", timeout=remaining(start, 58, time.monotonic()))
        state["exit"] = observed["State"]["ExitCode"]
        state["terminated"] = observed["State"]["Running"] is False
        state["inner_deadline_status"] = "timeout_exit" if state["exit"] == 124 else (
            "sigkill_or_deadline_unidentified" if state["exit"] == 137 else "not_indicated")
        state["timed_out"] |= state["exit"] == 124
    except (OSError, subprocess.SubprocessError, KeyError, ValueError) as exc:
        state["error_type"] = type(exc).__name__  # Never publish local paths from exception text.
    finally:
        try:
            if created:
                _api("DELETE", "containers/" + container_id + "?force=1", timeout=remaining(start, 59, time.monotonic()))
            after = _api("GET", "containers/json?all=1", timeout=remaining(start, 60, time.monotonic()))
            _docker_ps(output / "docker-after.txt", remaining(start, 60, time.monotonic()))
            publish(output / "docker-after.json", {"containers": after})
            state["containers_clean"] = not any("/" + name in c.get("Names", []) for c in after)
        except (OSError, subprocess.SubprocessError, ValueError):
            state["containers_clean"] = False
        state["elapsed_seconds"] = time.monotonic() - start
        captured = [p for p in output.iterdir() if p.is_file()]
        for role in ("native", "logs"):
            captured += [p for p in (output / role).rglob("*") if p.is_file()]
        state["artifacts"] = [artifact(p, output) for p in sorted(captured)]
        publish(output / "execution.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "capture", "judge"))
    parser.add_argument("--phase", choices=TESTS, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--assets", type=Path, default=Path("out/issue30/g0/assets"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list-tests", action="store_true")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        payload = request(args.phase, args.attempt, args.assets, args.out, uid=os.getuid(), gid=os.getgid())
        # Host mount paths stay private; the public plan uses role names only.
        payload["HostConfig"]["Binds"] = ["ASSETS:/assets:ro", "ATTEMPT:/capture:rw"]
        print(json.dumps(payload, indent=2))
        return 0
    if args.action == "capture":
        payload = None
        if args.approval:
            if args.list_tests or not args.before or not args.receipt:
                parser.error("hardware approval requires before/receipt and excludes listing")
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if subprocess.check_output(["git", "status", "--porcelain"]):
                parser.error("hardware capture requires a clean committed tree")
            env = json.loads(args.before.read_bytes())
            if env.get("boot") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
                parser.error("approval observations belong to a different boot")
            loaded = Path("/sys/module/tenstorrent/version")
            if not loaded.is_file() or loaded.read_text().strip() != env.get("kmd"):
                parser.error("loaded KMD does not match approval observations")
            receipt = json.loads(args.receipt.read_bytes())
            for source, expected in ((SOURCE_ROOT, SOURCE_COMMIT),
                                     (SOURCE_ROOT / "tt_metal/third_party/umd", UMD_COMMIT)):
                observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
                if observed != expected or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
                    parser.error("pinned source is stale or dirty")
            if (receipt.get("source_commit") != SOURCE_COMMIT
                    or receipt.get("runtime_library_sha256") != LIBRARY_SHA256):
                parser.error("receipt source/library pin mismatch")
            payload = board_request(args.phase, args.attempt, args.assets, args.out,
                                    approval=json.loads(args.approval.read_bytes()),
                                    environment=env, receipt=receipt, head=head)
        state = capture(args.phase, args.attempt, args.assets, args.out, listing=args.list_tests,
                        approved_payload=payload,
                        observations=json.loads(args.before.read_bytes()) if args.before else None)
        # A no-device test is never admitted as a real transfer, even on exit 0.
        return 0 if (args.list_tests and state["exit"] == 0 and state["terminated"]
                     and not state["timed_out"] and state["containers_clean"]
                     and state["elapsed_seconds"] <= 60) else 2
    if not args.before or not args.after or not args.review:
        parser.error("judge requires separate before/after observations and a review file")
    execution_path = args.out / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    review = json.loads(args.review.read_bytes())
    if execution.get("schema") != "issue30-execution-v1" or execution.get("attempt") != args.attempt:
        parser.error("execution schema/attempt mismatch")
    if (review.get("schema") != "issue30-review-v1" or review.get("attempt") != args.attempt
            or review.get("execution_sha256") != digest(execution_path.read_bytes())):
        parser.error("review is not bound to this immutable execution record")
    if execution.get("device_free") is not False:
        parser.error("device-free capture cannot qualify as board evidence")
    captured_before = json.loads((args.out / "observations-before.json").read_bytes())
    if captured_before != json.loads(args.before.read_bytes()):
        parser.error("before observations differ from the captured immutable copy")
    if args.phase == "B":
        watcher = args.out / "logs/generated/watcher/watcher.log"
        if (not watcher.is_file() or watcher.is_symlink() or watcher.stat().st_size == 0
                or review.get("watcher_sha256") != artifact(watcher, args.out)["sha256"]):
            parser.error("B requires a captured, reviewed Watcher log")
    execution = execution | {"watcher_clean": review.get("watcher_clean")}
    for field in ("before", "after"):
        path = getattr(args, field)
        if review.get(field + "_sha256") != digest(path.read_bytes()):
            parser.error("review observation fingerprint mismatch")
    result = evaluate(args.phase, args.attempt, json.loads(args.before.read_bytes()),
                      json.loads(args.after.read_bytes()), (args.out / "native/gtest.xml").read_bytes(),
                      (args.out / "stdout.bin").read_bytes(), (args.out / "stderr.bin").read_bytes(),
                      execution,
                      discovery=json.loads(args.discovery.read_bytes()) if args.discovery else None)
    publish(args.out / "result.json", result)
    return 0 if result["reason"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
