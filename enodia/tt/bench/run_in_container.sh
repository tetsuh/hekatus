#!/usr/bin/env bash
# Run the benchmark in the pinned toolchain container, with the environment
# that produced the numbers recorded beside them.
#
# The image is pinned by digest rather than by tag. design.md §2 warns that a
# firmware or version change has already moved the core count once, so a
# measurement whose toolchain cannot be named again later is not evidence.
#
# Usage: run_in_container.sh [output-directory] [-- extra runner arguments]
set -euo pipefail

IMAGE="${HEKATUS_TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:ead7b800bdb6bebb9425c377222314447c5b2052f6e8b1e3c9caa1818cb7d8c4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/out/bench}"
[ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
mkdir -p "${OUT_DIR}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ENV_JSON="${OUT_DIR}/env-${STAMP}.json"
POWER_CSV="${OUT_DIR}/power-${STAMP}.csv"
RESULTS="${OUT_DIR}/results-${STAMP}.json"

# --- environment, captured from the host where the board is visible ---------
python3 - "$ENV_JSON" "$IMAGE" <<'PY'
import json, subprocess, sys, datetime

out_path, image = sys.argv[1], sys.argv[2]


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:
        return f"<unavailable: {exc}>"


env = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "image": image,
    "kernel": run("uname -sr").strip(),
    "kmd_version": run("modinfo tenstorrent 2>/dev/null | awk '/^version:/{print $2}'").strip(),
    "tt_env_active_release": run(
        "tt-env status 2>/dev/null | awk '/Active release:/{print $3}'"
    ).strip(),
}

snapshot = run("tt-smi -s --snapshot_no_tty 2>/dev/null")
try:
    device = json.loads(snapshot)["device_info"][0]
    env["board"] = device["board_info"]
    env["firmware"] = device["firmwares"]
    env["limits"] = device["limits"]
except Exception as exc:
    env["board_snapshot_error"] = str(exc)

with open(out_path, "w") as fh:
    json.dump(env, fh, indent=2)
print(f"environment -> {out_path}")
PY

# --- board power and clock, sampled while the run proceeds ------------------
# This is also what settles the open power-limit question: whichever limit the
# board really enforces shows up here under sustained load.
(
  echo "timestamp_utc,power_w,aiclk_mhz,asic_temp_c"
  while true; do
    tt-smi -s --snapshot_no_tty 2>/dev/null | python3 -c '
import json, sys, datetime
try:
    t = json.load(sys.stdin)["device_info"][0]["telemetry"]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"{now},{t[\"power\"].strip()},{t[\"aiclk\"].strip()},{t[\"asic_temperature\"].strip()}")
except Exception:
    pass
' || true
    sleep 2
  done
) > "${POWER_CSV}" &
SAMPLER_PID=$!
trap 'kill "${SAMPLER_PID}" 2>/dev/null || true' EXIT

# --- the run itself ---------------------------------------------------------
docker run --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "${REPO_ROOT}:/work" \
  -v "${OUT_DIR}:/out" \
  -w /work \
  -e PYTHONPATH=/work \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc "python3 enodia/tt/bench/run_matmul.py --out /out/$(basename "${RESULTS}") --env-json /out/$(basename "${ENV_JSON}") $*"

kill "${SAMPLER_PID}" 2>/dev/null || true
echo
echo "results     -> ${RESULTS}"
echo "power trace -> ${POWER_CSV}"
