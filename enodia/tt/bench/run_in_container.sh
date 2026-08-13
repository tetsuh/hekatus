#!/usr/bin/env bash
# Run the benchmark in the pinned toolchain container, with the environment
# that produced the numbers recorded beside them.
#
# The image is pinned by digest rather than by tag. design.md §2 warns that a
# firmware or version change has already moved the core count once, so a
# measurement whose toolchain cannot be named again later is not evidence.
# An override is resolved to its digest where possible, and when it cannot
# be, the results record that they came from an unpinned image.
#
# Usage:
#   run_in_container.sh                          # default output directory
#   run_in_container.sh OUT_DIR                  # choose where results land
#   run_in_container.sh -- --iters 5             # arguments for the runner
#   run_in_container.sh OUT_DIR -- --iters 5
set -euo pipefail

IMAGE="${HEKATUS_TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:ead7b800bdb6bebb9425c377222314447c5b2052f6e8b1e3c9caa1818cb7d8c4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Arguments: an optional output directory, then "--", then runner arguments.
OUT_DIR="${REPO_ROOT}/out/bench"
if [[ $# -gt 0 && "$1" != "--" ]]; then
  OUT_DIR="$1"
  shift
fi
[[ "${1:-}" == "--" ]] && shift

mkdir -p "${OUT_DIR}"

# Resolve a tag to the digest it currently points at, so the recorded
# environment names one immutable toolchain rather than a moving one.
IMAGE_PINNED=0
case "${IMAGE}" in
  *@sha256:*) IMAGE_PINNED=1 ;;
  *)
    if RESOLVED="$(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE}" 2>/dev/null)" \
       && [[ -n "${RESOLVED}" ]]; then
      echo "resolved ${IMAGE} to ${RESOLVED}"
      IMAGE="${RESOLVED}"
      IMAGE_PINNED=1
    else
      echo "WARNING: ${IMAGE} is not digest-pinned and could not be resolved;" >&2
      echo "         results will be recorded as coming from an unpinned image." >&2
    fi
    ;;
esac

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ENV_JSON="${OUT_DIR}/env-${STAMP}.json"
POWER_CSV="${OUT_DIR}/power-${STAMP}.csv"
RESULTS="${OUT_DIR}/results-${STAMP}.json"

TELEMETRY="${REPO_ROOT}/enodia/tt/bench/telemetry.py"
PINNED_FLAG=()
[[ "${IMAGE_PINNED}" == "1" ]] && PINNED_FLAG=(--image-pinned)

python3 "${TELEMETRY}" capture-env --out "${ENV_JSON}" --image "${IMAGE}" "${PINNED_FLAG[@]}"

# Board power and clock while the run proceeds. This is what settles which
# power limit the board actually enforces under sustained load.
python3 "${TELEMETRY}" sample --out "${POWER_CSV}" --interval 2 &
SAMPLER_PID=$!

# Own both children if the wrapper is interrupted. The EXIT trap is disarmed
# after normal reaping, so it cannot act on a stale PID later.
cleanup_children() {
  local pid
  for pid in "${SAMPLER_PID:-}" "${DOCKER_PID:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SAMPLER_PID:-}" "${DOCKER_PID:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
cleanup_on_exit() {
  local status=$?
  trap - EXIT INT TERM HUP
  cleanup_children
  exit "${status}"
}
trap cleanup_on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

# Runner arguments are passed as separate arguments, never interpolated into
# a shell string: the wrapper must not turn a benchmark option into a command.
docker run --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "${REPO_ROOT}:/work" \
  -v "${OUT_DIR}:/out" \
  -w /work \
  -e PYTHONPATH=/work \
  --entrypoint /bin/bash \
  "${IMAGE}" -lc 'exec python3 "$0" --out "$1" --env-json "$2" "${@:3}"' \
  enodia/tt/bench/run_matmul.py \
  "/out/$(basename "${RESULTS}")" \
  "/out/$(basename "${ENV_JSON}")" \
  "$@" &
DOCKER_PID=$!

# A sampler that exits before Docker finishes means the run has no complete
# power trace. Wait for whichever process finishes first so that this failure
# cannot be hidden by the normal shutdown signal sent after Docker completes.
# Each status is captured on the line after its own `wait`: any command in
# between — an assignment included — replaces $? with its own success.
set +e
wait -n -p FINISHED "${SAMPLER_PID}" "${DOCKER_PID}"
FIRST_STATUS=$?

if [[ "${FINISHED}" == "${SAMPLER_PID}" ]]; then
  # The sampler stopped while the benchmark was still running, so the trace is
  # incomplete however the benchmark itself ends.
  SAMPLER_STATUS="${FIRST_STATUS}"
  echo "telemetry sampler exited before benchmark completion" >&2
  wait "${DOCKER_PID}"
  DOCKER_STATUS=$?
else
  DOCKER_STATUS="${FIRST_STATUS}"
  # The intentional shutdown: the benchmark finished, so the sampler has done
  # its job and is asked to stop.
  SAMPLER_STATUS=intentional
  kill "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}"
fi
set -e
trap - EXIT INT TERM HUP

# A failed benchmark is reported before a failed observer: a caller has to be
# able to tell a broken run from a broken measurement of a working one.

if [[ "${DOCKER_STATUS}" -ne 0 ]]; then
  exit "${DOCKER_STATUS}"
fi
if [[ "${SAMPLER_STATUS}" != intentional ]]; then
  echo "telemetry sampler failed (status ${SAMPLER_STATUS})" >&2
  exit 1
fi

echo
echo "results     -> ${RESULTS}"
echo "power trace -> ${POWER_CSV}"
