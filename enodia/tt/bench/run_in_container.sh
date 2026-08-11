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
trap 'kill "${SAMPLER_PID}" 2>/dev/null || true' EXIT

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
  "$@"

kill "${SAMPLER_PID}" 2>/dev/null || true
echo
echo "results     -> ${RESULTS}"
echo "power trace -> ${POWER_CSV}"
