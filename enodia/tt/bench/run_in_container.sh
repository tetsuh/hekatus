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
#   run_in_container.sh --pytest -m tt_device tests/test_newton_schulz_kernel.py
#
# `--pytest` is the only supported device-test entry point. It runs pytest in
# the digest-pinned image with HEKATUS_TT_DEVICE_TEST=1 and
# HEKATUS_TT_PINNED_CONTAINER=1; host pytest never opts into those tests.
set -euo pipefail

IMAGE="${HEKATUS_TT_IMAGE:-ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@sha256:5215587b1e3887f22f7dcd890c3ff4e23a58cd8e0beeb7569528b8ac2ccae621}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

TEST_MODE=0
if [[ "${1:-}" == "--pytest" ]]; then
  TEST_MODE=1
  shift
  PYTEST_ARGS=("$@")
  OUT_DIR="${HEKATUS_TT_TEST_OUT_DIR:-${REPO_ROOT}/out/bench-tests}"
else
  # Arguments: an optional output directory, then "--", then runner arguments.
  OUT_DIR="${REPO_ROOT}/out/bench"
  if [[ $# -gt 0 && "$1" != "--" ]]; then
    OUT_DIR="$1"
    shift
  fi
  [[ "${1:-}" == "--" ]] && shift
fi

mkdir -p "${OUT_DIR}"

# The default runs the benchmark.  A board-side Python probe can opt in with
# HEKATUS_TT_RUNNER; arguments after `--` are passed to that runner unchanged.
RUNNER="${HEKATUS_TT_RUNNER:-enodia/tt/bench/run_matmul.py}"
CONTAINER_TIMEOUT_S="${HEKATUS_TT_CONTAINER_TIMEOUT_S:-60}"
if ! [[ "${CONTAINER_TIMEOUT_S}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HEKATUS_TT_CONTAINER_TIMEOUT_S must be a positive integer" >&2
  exit 2
fi
CONTAINER_NAME="hekatus-bench-${$}-${RANDOM}"
DEVICE_NODE="${HEKATUS_TT_DEVICE_NODE:-/dev/tenstorrent/0}"
WATCHER_ENV=()
if [[ -n "${TT_METAL_WATCHER:-}" ]]; then
  WATCHER_ENV=(-e "TT_METAL_WATCHER=${TT_METAL_WATCHER}")
fi

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

# Own the children and the daemon-side container from before either is
# launched. Killing only the Docker client is insufficient: Docker can keep
# the board-side container running after an outer `timeout` or SSH disconnect.
# Installing the traps after a launch leaves a window in which a signal kills
# the wrapper and orphans the child it had just started; `${!:-}` closes the
# second one. It names the most recently started background job, exactly the
# child whose variable has not been assigned yet, and is empty before launch.
stop_container() {
  docker kill --signal KILL "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
cleanup_children() {
  local pid
  stop_container
  for pid in "${SAMPLER_PID:-}" "${DOCKER_PID:-}" "${!:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SAMPLER_PID:-}" "${DOCKER_PID:-}" "${!:-}"; do
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

# Board power and clock while the run proceeds. This is what settles which
# power limit the board actually enforces under sustained load.
python3 "${TELEMETRY}" sample --out "${POWER_CSV}" --interval 2 &
SAMPLER_PID=$!

# Runner arguments are passed as separate arguments, never interpolated into
# a shell string. The default runner also receives the sibling power trace
# path, so its JSON record names the same provenance that the wrapper writes.
# A custom probe is responsible for its own argument contract.
RUNNER_ARGS=("$@")
if [[ "${RUNNER}" == "enodia/tt/bench/run_matmul.py" ]]; then
  HAS_POWER_TRACE_ARG=0
  for argument in "${RUNNER_ARGS[@]}"; do
    if [[ "${argument}" == "--power-trace" ]]; then
      HAS_POWER_TRACE_ARG=1
      break
    fi
  done
  if [[ "${HAS_POWER_TRACE_ARG}" == "0" ]]; then
    RUNNER_ARGS+=(--power-trace "/out/$(basename "${POWER_CSV}")")
  fi
fi
# `timeout` is inside the wrapper, so losing an SSH session cannot leave the
# Docker client or the board-side container unbounded.
if [[ "${TEST_MODE}" == "1" ]]; then
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "${REPO_ROOT}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    -e HEKATUS_TT_DEVICE_TEST=1 \
    -e HEKATUS_TT_PINNED_CONTAINER=1 \
    "${WATCHER_ENV[@]}" \
    --entrypoint python3 \
    "${IMAGE}" -m pytest "${PYTEST_ARGS[@]}" &
elif [[ "${RUNNER}" == "enodia/tt/bench/run_matmul.py" ]]; then
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "${REPO_ROOT}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    "${WATCHER_ENV[@]}" \
    --entrypoint /bin/bash \
    "${IMAGE}" -lc 'exec python3 "$0" --out "$1" --env-json "$2" "${@:3}"' \
    "${RUNNER}" \
    "/out/$(basename "${RESULTS}")" \
    "/out/$(basename "${ENV_JSON}")" \
    "${RUNNER_ARGS[@]}" &
else
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v "${REPO_ROOT}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    "${WATCHER_ENV[@]}" \
    --entrypoint python3 \
    "${IMAGE}" "${RUNNER}" "$@" &
fi
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
  # A timeout normally invokes the EXIT trap only after this wait returns, but
  # explicitly kill once more for status 124.  This is idempotent after --rm
  # and covers Docker clients that do not propagate TERM to the named
  # container.
  stop_container
  exit "${DOCKER_STATUS}"
fi
if [[ "${SAMPLER_STATUS}" != intentional ]]; then
  echo "telemetry sampler failed (status ${SAMPLER_STATUS})" >&2
  exit 1
fi

echo
echo "results     -> ${RESULTS}"
echo "power trace -> ${POWER_CSV}"
