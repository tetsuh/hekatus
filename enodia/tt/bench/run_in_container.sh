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
RESIDENT_RUNNER_RELATIVE_PATH="enodia/tt/bench/run_resident.py"
MATMUL_RUNNER_RELATIVE_PATH="enodia/tt/bench/run_matmul.py"
ISSUE101_RUNNER_RELATIVE_PATH="tools/newton_schulz_issue101_combined.py"
ISSUE127_RUNNER_RELATIVE_PATH="tools/newton_schulz_issue127.py"
ISSUE137_RUNNER_RELATIVE_PATH="tools/issue137_device.py"

# Resolve runner paths against the host repository that is mounted at /work.
# The returned path is the only representation used by runner classifiers and
# is also the path passed to the container.
normalize_runner_path() {
  local runner="$1"
  local absolute_mount_path candidate normalized_root normalized relative
  if [[ -z "${runner}" || "${runner}" == *$'\n'* ]]; then
    echo "HEKATUS_TT_RUNNER must be nonempty and contain no newline" >&2
    return 2
  fi

  if [[ "${runner}" == /* ]]; then
    # Canonicalize absolute inputs first so duplicate-slash /work aliases are
    # treated like the path visible inside the container.
    if ! absolute_mount_path="$(realpath -m -- "${runner}" 2>/dev/null)" \
      || [[ -z "${absolute_mount_path}" ]]; then
      echo "cannot normalize HEKATUS_TT_RUNNER: ${runner}" >&2
      return 2
    fi
    case "${absolute_mount_path}" in
      /work)
        candidate="${REPO_ROOT}"
        ;;
      /work/*)
        candidate="${REPO_ROOT}/${absolute_mount_path#/work/}"
        ;;
      *)
        # Other absolute paths are host paths and are checked against the
        # canonical host repository below.
        candidate="${runner}"
        ;;
    esac
  else
    candidate="${REPO_ROOT}/${runner}"
  fi

  if ! normalized_root="$(realpath -m -- "${REPO_ROOT}" 2>/dev/null)" \
    || [[ -z "${normalized_root}" || "${normalized_root:0:1}" != "/" ]]; then
    echo "cannot normalize repository root for HEKATUS_TT_RUNNER" >&2
    return 2
  fi
  if ! normalized="$(realpath -m -- "${candidate}" 2>/dev/null)" \
    || [[ -z "${normalized}" || "${normalized:0:1}" != "/" || "${normalized}" == *$'\n'* ]]; then
    echo "cannot normalize HEKATUS_TT_RUNNER: ${runner}" >&2
    return 2
  fi

  case "${normalized}" in
    "${normalized_root}")
      relative="."
      ;;
    "${normalized_root}"/*)
      relative="${normalized#"${normalized_root}"/}"
      ;;
    *)
      echo "HEKATUS_TT_RUNNER resolves outside repository: ${runner}" >&2
      return 2
      ;;
  esac
  printf '%s\n' "${relative}"
}

RUNNER="${HEKATUS_TT_RUNNER:-${MATMUL_RUNNER_RELATIVE_PATH}}"
if ! RUNNER="$(normalize_runner_path "${RUNNER}")"; then
  exit 2
fi
CUSTOM_RUNNER=0
if [[ -n "${HEKATUS_TT_RUNNER:-}" ]]; then
  CUSTOM_RUNNER=1
fi

# Docker bind sources must be absolute host paths. `realpath -m` also handles
# an output directory whose parent does not exist yet; mkdir below creates it
# after this validation, before Docker is invoked.
normalize_host_mount_source() {
  local source="$1"
  local normalized
  if [[ -z "${source}" || "${source}" == *$'\n'* ]]; then
    echo "cannot normalize empty or newline-containing Docker bind source" >&2
    return 2
  fi
  if ! normalized="$(realpath -m -- "${source}" 2>/dev/null)" \
    || [[ -z "${normalized}" || "${normalized:0:1}" != "/" || "${normalized}" == *$'\n'* ]]; then
    echo "cannot normalize Docker bind source: ${source}" >&2
    return 2
  fi
  printf '%s\n' "${normalized}"
}

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

# The existing sampler is the default. Issue #104 can disable it for a
# diagnostic-only run or select a different positive interval without changing
# the benchmark or its kernel.
TELEMETRY_MODE="${HEKATUS_TT_TELEMETRY_MODE:-default}"
TELEMETRY_INTERVAL_S="${HEKATUS_TT_TELEMETRY_INTERVAL_S:-}"
case "${TELEMETRY_MODE}" in
  off)
    if [[ -n "${TELEMETRY_INTERVAL_S}" ]]; then
      echo "HEKATUS_TT_TELEMETRY_INTERVAL_S is not valid with sampler mode off" >&2
      exit 2
    fi
    ;;
  default)
    if [[ -n "${TELEMETRY_INTERVAL_S}" ]]; then
      echo "HEKATUS_TT_TELEMETRY_INTERVAL_S requires HEKATUS_TT_TELEMETRY_MODE=explicit" >&2
      exit 2
    fi
    TELEMETRY_INTERVAL_S=2
    ;;
  explicit)
    if [[ -z "${TELEMETRY_INTERVAL_S}" ]]; then
      echo "explicit sampler mode requires HEKATUS_TT_TELEMETRY_INTERVAL_S" >&2
      exit 2
    fi
    ;;
  *)
    echo "HEKATUS_TT_TELEMETRY_MODE must be off, default, or explicit" >&2
    exit 2
    ;;
esac

# The default runs the benchmark.  A board-side Python probe can opt in with
# HEKATUS_TT_RUNNER; arguments after `--` are passed to that runner unchanged.
RUNNER_ARGS=("$@")
TIMEOUT_EXPLICIT=0
if [[ -v HEKATUS_TT_CONTAINER_TIMEOUT_S ]]; then
  TIMEOUT_EXPLICIT=1
  CONTAINER_TIMEOUT_S="${HEKATUS_TT_CONTAINER_TIMEOUT_S}"
else
  CONTAINER_TIMEOUT_S=""
fi

validate_decimal_timeout() {
  local value="$1"
  local max_digits="$2"
  if [[ -z "${value}" || ! "${value}" =~ ^[0-9]+$ || "${value:0:1}" == "0" ]]; then
    echo "HEKATUS_TT_CONTAINER_TIMEOUT_S must be nonempty decimal digits with no leading zero" >&2
    exit 2
  fi
  if [[ "${#value}" -gt "${max_digits}" ]]; then
    echo "HEKATUS_TT_CONTAINER_TIMEOUT_S has too many digits" >&2
    exit 2
  fi
}

RESIDENT_RUNNER=0
RESIDENT_WATCHER_MODE=0
ISSUE127_RUNNER=0
ISSUE137_RUNNER=0
ISSUE127_DEFAULT_TIMEOUT_S="60"
ISSUE127_APPROVED_TIMEOUT_S="90"
ISSUE127_COOLDOWN_WAIT_CAP_S="600"
ISSUE127_COOLDOWN_BASELINE_SOURCE="measurement-1 pre-run power trace"
ISSUE127_COOLDOWN_BASELINE_POWER_W="47.0"
ISSUE127_COOLDOWN_BASELINE_TEMPERATURE_C="58.5"
ISSUE127_COOLDOWN_POWER_TOLERANCE_W="5.0"
ISSUE127_COOLDOWN_TEMPERATURE_TOLERANCE_C="3.0"
ISSUE127_COOLDOWN_POWER_LIMIT_W="52.0"
ISSUE127_COOLDOWN_TEMPERATURE_LIMIT_C="61.5"
RESIDENT_TIMEOUT_CAP_S="600"
RESIDENT_CONDITION="unloaded"
RESIDENT_REQUESTED_TIMEOUT=""
if [[ "${CUSTOM_RUNNER}" == "1" && "${RUNNER}" == "${RESIDENT_RUNNER_RELATIVE_PATH}" ]]; then
  RESIDENT_RUNNER=1
  EXPECT_CONDITION=0
  EXPECT_TIMEOUT=0
  for argument in "${RUNNER_ARGS[@]}"; do
    if [[ "${EXPECT_CONDITION}" == "1" ]]; then
      RESIDENT_CONDITION="${argument}"
      EXPECT_CONDITION=0
    elif [[ "${EXPECT_TIMEOUT}" == "1" ]]; then
      RESIDENT_REQUESTED_TIMEOUT="${argument}"
      EXPECT_TIMEOUT=0
    elif [[ "${argument}" == "--condition" ]]; then
      EXPECT_CONDITION=1
    elif [[ "${argument}" == --condition=* ]]; then
      RESIDENT_CONDITION="${argument#--condition=}"
    elif [[ "${argument}" == "--outer-timeout-seconds" ]]; then
      EXPECT_TIMEOUT=1
    elif [[ "${argument}" == --outer-timeout-seconds=* ]]; then
      RESIDENT_REQUESTED_TIMEOUT="${argument#--outer-timeout-seconds=}"
    fi
  done
  CLI_WATCHER_MODE=0
  for argument in "${RUNNER_ARGS[@]}"; do
    if [[ "${argument}" == "--watcher" ]]; then
      CLI_WATCHER_MODE=1
      break
    fi
  done
  ENV_WATCHER_MODE=0
  if [[ -v TT_METAL_WATCHER ]]; then
    if [[ "${TT_METAL_WATCHER}" != "1" ]]; then
      echo "TT_METAL_WATCHER must be unset or exactly 1 for resident runs" >&2
      exit 2
    fi
    ENV_WATCHER_MODE=1
  fi
  if [[ "${CLI_WATCHER_MODE}" != "${ENV_WATCHER_MODE}" ]]; then
    echo "resident --watcher and TT_METAL_WATCHER must select the same mode" >&2
    exit 2
  fi
  RESIDENT_WATCHER_MODE="${CLI_WATCHER_MODE}"
  if [[ "${RESIDENT_WATCHER_MODE}" == "1" ]]; then
    RESIDENT_TIMEOUT_CAP_S="60"
  elif [[ "${RESIDENT_CONDITION}" == "newton_schulz" \
          && "${RESIDENT_REQUESTED_TIMEOUT}" == "800" ]]; then
    # Compute-loaded Issue #118 runs may opt into the separately recorded
    # 800-second ceiling; all other resident invocations retain 600 seconds.
    RESIDENT_TIMEOUT_CAP_S="800"
  fi
  if [[ "${TIMEOUT_EXPLICIT}" == "0" ]]; then
    CONTAINER_TIMEOUT_S="${RESIDENT_TIMEOUT_CAP_S}"
  fi
  validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "${#RESIDENT_TIMEOUT_CAP_S}"
  if [[ "${#CONTAINER_TIMEOUT_S}" == "${#RESIDENT_TIMEOUT_CAP_S}" \
        && "${CONTAINER_TIMEOUT_S}" > "${RESIDENT_TIMEOUT_CAP_S}" ]]; then
    echo "resident container timeout ${CONTAINER_TIMEOUT_S}s exceeds the ${RESIDENT_TIMEOUT_CAP_S}s approved cap" >&2
    exit 2
  fi
elif [[ "${CUSTOM_RUNNER}" == "1" && "${RUNNER}" == "${ISSUE137_RUNNER_RELATIVE_PATH}" ]]; then
  ISSUE137_RUNNER=1
  ISSUE137_REQUESTED_TIMEOUT=""
  ISSUE137_MODE=""
  ISSUE137_ARGUMENT_INDEX=0
  while [[ "${ISSUE137_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${ISSUE137_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --mode)
        if [[ "${ISSUE137_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--mode requires a value" >&2
          exit 2
        fi
        if [[ -n "${ISSUE137_MODE}" ]]; then
          echo "Issue #137 mode may be supplied only once" >&2
          exit 2
        fi
        ISSUE137_MODE="${RUNNER_ARGS[$((ISSUE137_ARGUMENT_INDEX + 1))]}"
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 2))
        continue
        ;;
      --mode=*)
        if [[ -n "${ISSUE137_MODE}" ]]; then
          echo "Issue #137 mode may be supplied only once" >&2
          exit 2
        fi
        ISSUE137_MODE="${argument#--mode=}"
        ;;
      --outer-timeout-seconds)
        if [[ "${ISSUE137_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--outer-timeout-seconds requires a value" >&2
          exit 2
        fi
        if [[ -n "${ISSUE137_REQUESTED_TIMEOUT}" ]]; then
          echo "Issue #137 outer timeout may be supplied only once" >&2
          exit 2
        fi
        ISSUE137_REQUESTED_TIMEOUT="${RUNNER_ARGS[$((ISSUE137_ARGUMENT_INDEX + 1))]}"
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 2))
        continue
        ;;
      --outer-timeout-seconds=*)
        if [[ -n "${ISSUE137_REQUESTED_TIMEOUT}" ]]; then
          echo "Issue #137 outer timeout may be supplied only once" >&2
          exit 2
        fi
        ISSUE137_REQUESTED_TIMEOUT="${argument#--outer-timeout-seconds=}"
        ;;
    esac
    ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 1))
  done
  case "${ISSUE137_MODE}" in
    smoke) ISSUE137_EXPECTED_TIMEOUT="70" ;;
    long) ISSUE137_EXPECTED_TIMEOUT="170" ;;
    *)
      echo "Issue #137 requires --mode smoke or --mode long" >&2
      exit 2
      ;;
  esac
  if [[ "${ISSUE137_REQUESTED_TIMEOUT}" != "${ISSUE137_EXPECTED_TIMEOUT}" ]]; then
    echo "Issue #137 ${ISSUE137_MODE} mode requires a ${ISSUE137_EXPECTED_TIMEOUT}s guard" >&2
    exit 2
  fi
  if [[ "${TIMEOUT_EXPLICIT}" == "1" ]]; then
    validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "3"
    if [[ "${CONTAINER_TIMEOUT_S}" != "${ISSUE137_REQUESTED_TIMEOUT}" ]]; then
      echo "Issue #137 container timeout must match the fixed runner guard" >&2
      exit 2
    fi
  else
    CONTAINER_TIMEOUT_S="${ISSUE137_REQUESTED_TIMEOUT}"
  fi
  validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "3"
  if [[ "${RUNNER_ARGS[*]}" == *"--calibrat"* ]]; then
    echo "Issue #137 does not accept calibration arguments" >&2
    exit 2
  fi
elif [[ "${CUSTOM_RUNNER}" == "1" && "${RUNNER}" == "${ISSUE127_RUNNER_RELATIVE_PATH}" ]]; then
  # Issue #127 defaults to a 60-second run budget.  Only the explicit owner-
  # approved scope plus --run-timeout-seconds 90 may raise the Docker timeout.
  ISSUE127_RUNNER=1
  if [[ "${TELEMETRY_MODE}" == "off" ]]; then
    echo "Issue #127 requires telemetry sampler mode default or explicit; sampler-off runs are rejected" >&2
    exit 2
  fi
  ISSUE127_RUN_NUMBER=""
  ISSUE127_COOLDOWN_SECONDS=""
  ISSUE127_EXPECTED_AICLK=""
  ISSUE127_RUN_TIMEOUT_SECONDS=""
  ISSUE127_RUN_SCOPE=""
  ISSUE127_COOLDOWN_BOUNDS_SUPPLIED=0
  ISSUE127_ARGUMENT_INDEX=0
  while [[ "${ISSUE127_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${ISSUE127_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --run-number)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--run-number requires a value" >&2
          exit 2
        fi
        ISSUE127_RUN_NUMBER="${RUNNER_ARGS[$((ISSUE127_ARGUMENT_INDEX + 1))]}"
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --run-number=*)
        ISSUE127_RUN_NUMBER="${argument#--run-number=}"
        ;;
      --run-timeout-seconds)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--run-timeout-seconds requires a value" >&2
          exit 2
        fi
        if [[ -n "${ISSUE127_RUN_TIMEOUT_SECONDS}" ]]; then
          echo "Issue #127 run timeout may be supplied only once" >&2
          exit 2
        fi
        ISSUE127_RUN_TIMEOUT_SECONDS="${RUNNER_ARGS[$((ISSUE127_ARGUMENT_INDEX + 1))]}"
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --run-timeout-seconds=*)
        if [[ -n "${ISSUE127_RUN_TIMEOUT_SECONDS}" ]]; then
          echo "Issue #127 run timeout may be supplied only once" >&2
          exit 2
        fi
        ISSUE127_RUN_TIMEOUT_SECONDS="${argument#--run-timeout-seconds=}"
        ;;
      --run-scope)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--run-scope requires a value" >&2
          exit 2
        fi
        if [[ -n "${ISSUE127_RUN_SCOPE}" ]]; then
          echo "Issue #127 run scope may be supplied only once" >&2
          exit 2
        fi
        ISSUE127_RUN_SCOPE="${RUNNER_ARGS[$((ISSUE127_ARGUMENT_INDEX + 1))]}"
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --run-scope=*)
        if [[ -n "${ISSUE127_RUN_SCOPE}" ]]; then
          echo "Issue #127 run scope may be supplied only once" >&2
          exit 2
        fi
        ISSUE127_RUN_SCOPE="${argument#--run-scope=}"
        ;;
      --cooldown-wait-seconds)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--cooldown-wait-seconds requires a value" >&2
          exit 2
        fi
        ISSUE127_COOLDOWN_SECONDS="${RUNNER_ARGS[$((ISSUE127_ARGUMENT_INDEX + 1))]}"
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --cooldown-wait-seconds=*)
        ISSUE127_COOLDOWN_SECONDS="${argument#--cooldown-wait-seconds=}"
        ;;
      --expected-aiclk-mhz)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--expected-aiclk-mhz requires a value" >&2
          exit 2
        fi
        ISSUE127_EXPECTED_AICLK="${RUNNER_ARGS[$((ISSUE127_ARGUMENT_INDEX + 1))]}"
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --expected-aiclk-mhz=*)
        ISSUE127_EXPECTED_AICLK="${argument#--expected-aiclk-mhz=}"
        ;;
      --cooldown-start-utc|--cooldown-end-utc|--cooldown-start-utc=*|--cooldown-end-utc=*)
        ISSUE127_COOLDOWN_BOUNDS_SUPPLIED=1
        ;;
    esac
    ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
  done
  if [[ "${ISSUE127_RUN_NUMBER}" == "2" ]]; then
    if [[ -n "${ISSUE127_COOLDOWN_SECONDS}" ]]; then
      if [[ ! "${ISSUE127_COOLDOWN_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ \
            || "${ISSUE127_COOLDOWN_SECONDS}" =~ ^0+([.]0+)?$ ]]; then
        echo "Issue #127 run 2 requires a positive numeric cooldown wait" >&2
        exit 2
      fi
      if ! awk -v value="${ISSUE127_COOLDOWN_SECONDS}" \
          'BEGIN { exit !(value > 0 && value <= 600) }'; then
        echo "Issue #127 cooldown wait must be at most 600 seconds" >&2
        exit 2
      fi
    fi
    # AICLK is accepted only as legacy provenance input.  The corrected gate
    # below never compares it with a target value.
    if [[ "${ISSUE127_COOLDOWN_BOUNDS_SUPPLIED}" == "1" ]]; then
      echo "Issue #127 cooldown UTC bounds are wrapper-owned" >&2
      exit 2
    fi
  elif [[ "${ISSUE127_RUN_NUMBER}" != "1" ]]; then
    echo "Issue #127 requires --run-number 1 or 2" >&2
    exit 2
  fi
  ISSUE127_EFFECTIVE_TIMEOUT_S="${ISSUE127_DEFAULT_TIMEOUT_S}"
  if [[ -z "${ISSUE127_RUN_TIMEOUT_SECONDS}" && -n "${ISSUE127_RUN_SCOPE}" ]]; then
    echo "Issue #127 approved run scope requires --run-timeout-seconds 90" >&2
    exit 2
  fi
  if [[ -n "${ISSUE127_RUN_TIMEOUT_SECONDS}" ]]; then
    if [[ ! "${ISSUE127_RUN_TIMEOUT_SECONDS}" =~ ^90([.]0+)?$ ]]; then
      if [[ "${ISSUE127_RUN_TIMEOUT_SECONDS}" =~ ^60([.]0+)?$ && -z "${ISSUE127_RUN_SCOPE}" ]]; then
        ISSUE127_EFFECTIVE_TIMEOUT_S="${ISSUE127_DEFAULT_TIMEOUT_S}"
      else
        echo "Issue #127 permits 90 seconds only for an approved run scope" >&2
        exit 2
      fi
    else
      case "${ISSUE127_RUN_SCOPE}" in
        confirmation|measurement-1)
          if [[ "${ISSUE127_RUN_NUMBER}" != "1" ]]; then
            echo "Issue #127 run scope ${ISSUE127_RUN_SCOPE} requires --run-number 1" >&2
            exit 2
          fi
          ;;
        measurement-2|measurement-2-rerun)
          if [[ "${ISSUE127_RUN_NUMBER}" != "2" ]]; then
            echo "Issue #127 run scope ${ISSUE127_RUN_SCOPE} requires --run-number 2" >&2
            exit 2
          fi
          ;;
        *)
          echo "Issue #127 run scope is not an approved target" >&2
          exit 2
          ;;
      esac
      ISSUE127_EFFECTIVE_TIMEOUT_S="${ISSUE127_APPROVED_TIMEOUT_S}"
    fi
  elif [[ -n "${ISSUE127_RUN_SCOPE}" ]]; then
    echo "Issue #127 approved run scope requires --run-timeout-seconds 90" >&2
    exit 2
  fi
  if [[ "${TIMEOUT_EXPLICIT}" == "0" ]]; then
    CONTAINER_TIMEOUT_S="${ISSUE127_EFFECTIVE_TIMEOUT_S}"
  else
    validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "2"
    if [[ "${CONTAINER_TIMEOUT_S}" != "${ISSUE127_EFFECTIVE_TIMEOUT_S}" ]]; then
      echo "Issue #127 container timeout must match the approved run timeout" >&2
      exit 2
    fi
  fi
  validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "2"
else
  # Non-resident wrapper users retain their historical timeout range, but the
  # input is still bounded before GNU timeout or any shell arithmetic sees it.
  if [[ "${TIMEOUT_EXPLICIT}" == "0" ]]; then
    CONTAINER_TIMEOUT_S="900"
  fi
  validate_decimal_timeout "${CONTAINER_TIMEOUT_S}" "9"
fi

if ! HUGEPAGES_SOURCE="$(normalize_host_mount_source /dev/hugepages-1G)"; then
  exit 2
fi
if ! REPO_ROOT_SOURCE="$(normalize_host_mount_source "${REPO_ROOT}")"; then
  exit 2
fi
if ! OUT_DIR="$(normalize_host_mount_source "${OUT_DIR}")"; then
  exit 2
fi
mkdir -p "${OUT_DIR}"
CONTAINER_NAME="hekatus-bench-${$}-${RANDOM}"
DEVICE_NODE="${HEKATUS_TT_DEVICE_NODE:-/dev/tenstorrent/0}"
# The host-side telemetry process must resolve the same effective node that
# Docker receives, including the default when the caller did not export it.
export HEKATUS_TT_DEVICE_NODE="${DEVICE_NODE}"
WATCHER_ENV=()
if [[ -n "${TT_METAL_WATCHER:-}" ]]; then
  WATCHER_ENV=(-e "TT_METAL_WATCHER=${TT_METAL_WATCHER}")
fi

# Resolve a tag to the digest it currently points at, so the recorded
# environment names one immutable toolchain rather than a moving one.
IMAGE_PINNED=0
image_is_digest_pinned() {
  [[ "$1" =~ @sha256:[0-9a-f]{64}$ ]]
}
case "${IMAGE}" in
  *@sha256:*)
    if image_is_digest_pinned "${IMAGE}"; then
      IMAGE_PINNED=1
    fi
    ;;
  *)
    if RESOLVED="$(docker image inspect --format '{{index .RepoDigests 0}}' "${IMAGE}" 2>/dev/null)" \
       && image_is_digest_pinned "${RESOLVED}"; then
      echo "resolved ${IMAGE} to ${RESOLVED}"
      IMAGE="${RESOLVED}"
      IMAGE_PINNED=1
    elif [[ "${RESIDENT_RUNNER}" == "1" || "${ISSUE137_RUNNER}" == "1" ]]; then
      echo "resident/Issue #137 runner requires HEKATUS_TT_IMAGE to resolve to @sha256:<64 lowercase hex>" >&2
      exit 2
    elif [[ "${CUSTOM_RUNNER}" == "1" \
            && "${RUNNER}" == "${ISSUE101_RUNNER_RELATIVE_PATH}" ]]; then
      echo "Issue #101 runner requires HEKATUS_TT_IMAGE to resolve to @sha256:<64 lowercase hex>" >&2
      exit 2
    else
      echo "WARNING: ${IMAGE} is not digest-pinned and could not be resolved;" >&2
      echo "         results will be recorded as coming from an unpinned image." >&2
    fi
    ;;
esac

if [[ "${RESIDENT_RUNNER}" == "1" && "${IMAGE_PINNED}" != "1" ]]; then
  echo "resident runner requires a verified digest-pinned image" >&2
  exit 2
fi
if [[ "${ISSUE137_RUNNER}" == "1" && "${IMAGE_PINNED}" != "1" ]]; then
  echo "Issue #137 runner requires a verified digest-pinned image" >&2
  exit 2
fi
if [[ "${CUSTOM_RUNNER}" == "1" \
      && "${RUNNER}" == "${ISSUE101_RUNNER_RELATIVE_PATH}" \
      && "${IMAGE_PINNED}" != "1" ]]; then
  echo "Issue #101 runner requires a verified digest-pinned image" >&2
  exit 2
fi

if [[ "${TEST_MODE}" == "1" && "${IMAGE_PINNED}" != "1" ]]; then
  echo "--pytest requires a digest-pinned HEKATUS_TT_IMAGE" >&2
  exit 2
fi

# One identity is generated before any artifact is opened.  The nanosecond,
# process, and shell-random suffix make reuse collisions fail-safe rather than
# silently overwriting a prior session's provenance.
RUN_ID="$(date -u +%Y%m%dT%H%M%S%N)$$${RANDOM}Z"
export HEKATUS_TT_RUN_ID="${RUN_ID}"
ENV_JSON="${OUT_DIR}/env-${RUN_ID}.json"
POWER_CSV="${OUT_DIR}/power-${RUN_ID}.csv"
RESULTS="${OUT_DIR}/results-${RUN_ID}.json"
RESULTS_CONTAINER="/out/$(basename "${RESULTS}")"
RESULT_PATH="${RESULTS}"
RUNNER_RESULT_CONTAINER_PATH=""
RUNNER_RESULT_HOST_PATH=""
if [[ "${CUSTOM_RUNNER}" == "1" ]]; then
  # Keep the historical one-run path, but never overwrite it when an output
  # directory is reused.  The run-specific fallback is passed into the
  # container and is not embedded as a host path in the runner record.
  if [[ -e "${OUT_DIR}/runner-result.json" ]]; then
    RUNNER_RESULT_CONTAINER_PATH="/out/runner-result-${RUN_ID}.json"
    RUNNER_RESULT_HOST_PATH="${OUT_DIR}/runner-result-${RUN_ID}.json"
  else
    RUNNER_RESULT_CONTAINER_PATH="/out/runner-result.json"
    RUNNER_RESULT_HOST_PATH="${OUT_DIR}/runner-result.json"
  fi
  RESULTS_CONTAINER="${RUNNER_RESULT_CONTAINER_PATH}"
  RESULT_PATH="${RUNNER_RESULT_HOST_PATH}"
fi

TELEMETRY="${REPO_ROOT}/enodia/tt/bench/telemetry.py"
PINNED_FLAG=()
[[ "${IMAGE_PINNED}" == "1" ]] && PINNED_FLAG=(--image-pinned)
RUNNER_RESULT_ENV=()
if [[ "${CUSTOM_RUNNER}" == "1" ]]; then
  RUNNER_RESULT_ENV=(-e "HEKATUS_TT_RESULT_PATH=${RUNNER_RESULT_CONTAINER_PATH}")
fi

CAPTURE_SAMPLER_ARGS=(--sampler-mode "${TELEMETRY_MODE}")
if [[ "${TELEMETRY_MODE}" != "off" ]]; then
  CAPTURE_SAMPLER_ARGS+=(--sampler-interval "${TELEMETRY_INTERVAL_S}")
fi
python3 "${TELEMETRY}" capture-env --out "${ENV_JSON}" --image "${IMAGE}" \
  "${PINNED_FLAG[@]}" "${CAPTURE_SAMPLER_ARGS[@]}"
TELEMETRY_ENV=(-e "HEKATUS_TT_TELEMETRY_MODE=${TELEMETRY_MODE}")
if [[ "${TELEMETRY_MODE}" != "off" ]]; then
  TELEMETRY_ENV+=(-e "HEKATUS_TT_TELEMETRY_INTERVAL_S=${TELEMETRY_INTERVAL_S}")
fi

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
      kill -TERM "${pid}" 2>/dev/null || true
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
# power limit the board actually enforces under sustained load. Sampler-off is
# intentionally trace-free and is used only as a diagnostic run.
SAMPLER_PID=""
if [[ "${TELEMETRY_MODE}" != "off" ]]; then
  python3 "${TELEMETRY}" sample --out "${POWER_CSV}" --interval "${TELEMETRY_INTERVAL_S}" &
  SAMPLER_PID=$!
fi

# Issue #127 run 2 cooldown is deliberately outside the Docker timeout.  The
# sampler is already running.  Wait for an idle power/temperature sample rather
# than requiring the card to return to a particular AICLK; AICLK is provenance
# only.  A lower cap can be selected by board-free wrapper tests, but production
# can never exceed the fixed 600-second cap.
if [[ "${ISSUE127_RUNNER}" == "1" && "${ISSUE127_RUN_NUMBER}" == "2" ]]; then
  COOLDOWN_CAP_S="${HEKATUS_TT_ISSUE127_COOLDOWN_CAP_SECONDS:-${ISSUE127_COOLDOWN_WAIT_CAP_S}}"
  if [[ ! "${COOLDOWN_CAP_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || ! awk -v value="${COOLDOWN_CAP_S}" \
          'BEGIN { exit !(value > 0 && value <= 600) }'; then
    echo "Issue #127 cooldown wait cap must be positive and at most 600 seconds" >&2
    exit 2
  fi

  issue127_latest_sample() {
    if [[ ! -f "${POWER_CSV}" ]]; then
      return 0
    fi
    LC_ALL=C awk -F, '
      NR == 1 { next }
      NF >= 4 \
        && $2 ~ /^[+-]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ \
        && $3 ~ /^[+-]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ \
        && $4 ~ /^[+-]?[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/ {
          timestamp = $1
          power = $2
          aiclk = $3
          temperature = $4
          found = 1
        }
      END {
        if (found) {
          printf "%.17g|%.17g|%.17g|%s\n", power, aiclk, temperature, timestamp
        }
      }
    ' "${POWER_CSV}" 2>/dev/null
  }

  issue127_timestamp_is_safe() {
    [[ "${1}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]([.][0-9]{1,9}|,[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$ ]]
  }

  issue127_timestamp_to_ns() {
    local timestamp="${1}"
    local parsed
    if ! issue127_timestamp_is_safe "${timestamp}"; then
      return 1
    fi
    if ! parsed="$(date -u --date="${timestamp}" +%s%N 2>/dev/null)" \
      || [[ ! "${parsed}" =~ ^-?[0-9]{1,19}$ ]]; then
      return 1
    fi
    printf '%s\n' "${parsed}"
  }

  issue127_write_cooldown_failure() {
    local failure_path="${OUT_DIR}/cooldown-gate-failure-${RUN_ID}.json"
    local run_scope_json="null"
    if [[ -n "${ISSUE127_RUN_SCOPE}" ]]; then
      run_scope_json="\"${ISSUE127_RUN_SCOPE}\""
    fi
    cat > "${failure_path}" <<EOF
{
  "issue": "#127",
  "run_number": 2,
  "run_scope": ${run_scope_json},
  "status": "failed",
  "failure_type": "cooldown_gate_failure",
  "measurement_started": false,
  "baseline_source": "${ISSUE127_COOLDOWN_BASELINE_SOURCE}",
  "baseline_power_w": ${ISSUE127_COOLDOWN_BASELINE_POWER_W},
  "baseline_temperature_c": ${ISSUE127_COOLDOWN_BASELINE_TEMPERATURE_C},
  "power_tolerance_w": ${ISSUE127_COOLDOWN_POWER_TOLERANCE_W},
  "temperature_tolerance_c": ${ISSUE127_COOLDOWN_TEMPERATURE_TOLERANCE_C},
  "power_limit_w": ${ISSUE127_COOLDOWN_POWER_LIMIT_W},
  "temperature_limit_c": ${ISSUE127_COOLDOWN_TEMPERATURE_LIMIT_C},
  "wait_cap_seconds": 600.0,
  "wait_performed": true,
  "wait_start_utc": "${COOLDOWN_START_UTC}",
  "wait_end_utc": "${COOLDOWN_END_UTC}",
  "actual_wait_seconds": ${COOLDOWN_ACTUAL_WAIT_SECONDS},
  "gate_result": "fail",
  "gate_verified": false,
  "aiclk_role": "provenance_only",
  "observed_power_w": ${COOLDOWN_OBSERVED_POWER_W:-null},
  "observed_temperature_c": ${COOLDOWN_OBSERVED_TEMPERATURE_C:-null},
  "observed_idle_aiclk_mhz": ${COOLDOWN_OBSERVED_AICLK_MHZ:-null},
  "observed_sample_timestamp_utc": ${COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_JSON}
}
EOF
  }

  COOLDOWN_START_UTC="$(date -u --iso-8601=ns)"
  if ! COOLDOWN_START_NS="$(issue127_timestamp_to_ns "${COOLDOWN_START_UTC}")"; then
    echo "Issue #127 cooldown start timestamp could not be parsed; refusing to start cooldown gate" >&2
    exit 2
  fi
  COOLDOWN_CAP_NS="$(awk -v start="${COOLDOWN_START_NS}" -v cap="${COOLDOWN_CAP_S}" \
    'BEGIN { printf "%.0f", start + cap * 1000000000 }')"
  COOLDOWN_GATE_PASSED=0
  COOLDOWN_GATE_REASON="cooldown gate did not pass before the wait cap"
  COOLDOWN_OBSERVED_POWER_W=""
  COOLDOWN_OBSERVED_TEMPERATURE_C=""
  COOLDOWN_OBSERVED_AICLK_MHZ=""
  COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_UTC=""
  COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_JSON="null"
  while :; do
    COOLDOWN_NOW_NS="$(date +%s%N)"
    if [[ "${COOLDOWN_NOW_NS}" -ge "${COOLDOWN_CAP_NS}" ]]; then
      break
    fi
    COOLDOWN_SAMPLE="$(issue127_latest_sample)"
    if [[ -n "${COOLDOWN_SAMPLE}" ]]; then
      IFS='|' read -r COOLDOWN_OBSERVED_POWER_W COOLDOWN_OBSERVED_AICLK_MHZ \
        COOLDOWN_OBSERVED_TEMPERATURE_C COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_UTC \
        <<< "${COOLDOWN_SAMPLE}"
      COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_JSON="null"
      if issue127_timestamp_is_safe "${COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_UTC}"; then
        # The strict ISO shape contains no JSON-significant characters beyond
        # the quotes added here, so malformed CSV data cannot escape the record.
        COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_JSON="\"${COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_UTC}\""
      fi
      if ! COOLDOWN_SAMPLE_TIMESTAMP_NS="$(issue127_timestamp_to_ns \
          "${COOLDOWN_OBSERVED_SAMPLE_TIMESTAMP_UTC}")"; then
        COOLDOWN_GATE_REASON="cooldown sample timestamp was malformed"
      elif [[ "${COOLDOWN_SAMPLE_TIMESTAMP_NS}" -lt "${COOLDOWN_START_NS}" ]]; then
        COOLDOWN_GATE_REASON="cooldown sample timestamp predates cooldown start"
      elif [[ "${COOLDOWN_SAMPLE_TIMESTAMP_NS}" -gt "${COOLDOWN_NOW_NS}" ]]; then
        COOLDOWN_GATE_REASON="cooldown sample timestamp is after the current wait time"
      elif awk -v power="${COOLDOWN_OBSERVED_POWER_W}" \
          -v temperature="${COOLDOWN_OBSERVED_TEMPERATURE_C}" \
          'BEGIN { exit !(power <= 52.0 && temperature <= 61.5) }'; then
        COOLDOWN_GATE_PASSED=1
        break
      else
        COOLDOWN_GATE_REASON="observed power/temperature exceeded the cooldown gate"
      fi
    fi
    if ! kill -0 "${SAMPLER_PID}" 2>/dev/null; then
      COOLDOWN_GATE_REASON="telemetry sampler exited before the cooldown gate passed"
      break
    fi
    COOLDOWN_REMAINING_S="$(awk -v deadline="${COOLDOWN_CAP_NS}" -v now="${COOLDOWN_NOW_NS}" \
      'BEGIN { remaining = (deadline - now) / 1000000000; printf "%.6f", remaining }')"
    if ! awk -v remaining="${COOLDOWN_REMAINING_S}" \
        'BEGIN { exit !(remaining > 0) }'; then
      break
    fi
    COOLDOWN_SLEEP_S="$(awk -v remaining="${COOLDOWN_REMAINING_S}" \
      'BEGIN { print (remaining < 0.25 ? remaining : 0.25) }')"
    sleep "${COOLDOWN_SLEEP_S}"
  done
  COOLDOWN_END_NS="$(date +%s%N)"
  COOLDOWN_END_UTC="$(date -u --iso-8601=ns)"
  COOLDOWN_ACTUAL_WAIT_SECONDS="$(awk -v start="${COOLDOWN_START_NS}" -v end="${COOLDOWN_END_NS}" \
    'BEGIN { printf "%.9f", (end - start) / 1000000000 }')"
  if [[ "${COOLDOWN_GATE_PASSED}" != "1" ]]; then
    issue127_write_cooldown_failure
    echo "Issue #127 cooldown gate failure: ${COOLDOWN_GATE_REASON}; measurement was not started" >&2
    echo "Issue #127 cooldown gate failure record -> ${OUT_DIR}/cooldown-gate-failure-${RUN_ID}.json" >&2
    exit 1
  fi
  # Replace the historical requested-duration argument with the fixed contract
  # cap so the runner persists one unambiguous 600-second wait bound.
  ISSUE127_FILTERED_ARGS=()
  ISSUE127_ARGUMENT_INDEX=0
  while [[ "${ISSUE127_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${ISSUE127_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --cooldown-wait-seconds)
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --cooldown-wait-seconds=*)
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
        continue
        ;;
      *)
        ISSUE127_FILTERED_ARGS+=("${argument}")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
        ;;
    esac
  done
  RUNNER_ARGS=("${ISSUE127_FILTERED_ARGS[@]}")
  RUNNER_ARGS+=(
    --cooldown-wait-seconds "${ISSUE127_COOLDOWN_WAIT_CAP_S}"
    --cooldown-start-utc "${COOLDOWN_START_UTC}"
    --cooldown-end-utc "${COOLDOWN_END_UTC}"
    --cooldown-baseline-source "${ISSUE127_COOLDOWN_BASELINE_SOURCE}"
    --cooldown-baseline-power-w "${ISSUE127_COOLDOWN_BASELINE_POWER_W}"
    --cooldown-baseline-temperature-c "${ISSUE127_COOLDOWN_BASELINE_TEMPERATURE_C}"
    --cooldown-power-tolerance-w "${ISSUE127_COOLDOWN_POWER_TOLERANCE_W}"
    --cooldown-temperature-tolerance-c "${ISSUE127_COOLDOWN_TEMPERATURE_TOLERANCE_C}"
    --cooldown-observed-power-w "${COOLDOWN_OBSERVED_POWER_W}"
    --cooldown-observed-temperature-c "${COOLDOWN_OBSERVED_TEMPERATURE_C}"
    --cooldown-observed-aiclk-mhz "${COOLDOWN_OBSERVED_AICLK_MHZ}"
    --cooldown-gate-result pass
    --cooldown-gate-verified
  )
fi

# Runner arguments are passed as separate arguments, never interpolated into
# a shell string. The default runner also receives the sibling power trace
# path, so its JSON record names the same provenance that the wrapper writes.
# A custom probe is responsible for its own argument contract.
if [[ "${RUNNER}" == "${MATMUL_RUNNER_RELATIVE_PATH}" ]]; then
  HAS_POWER_TRACE_ARG=0
  for argument in "${RUNNER_ARGS[@]}"; do
    if [[ "${argument}" == "--power-trace" ]]; then
      HAS_POWER_TRACE_ARG=1
      break
    fi
  done
  if [[ "${HAS_POWER_TRACE_ARG}" == "0" && "${TELEMETRY_MODE}" != "off" ]]; then
    RUNNER_ARGS+=(--power-trace "/out/$(basename "${POWER_CSV}")")
  fi
elif [[ "${RUNNER}" == *"enodia/tt/bench/run_resident.py" ]]; then
  # Resident records store sibling filenames, and the wrapper owns both
  # run-specific paths.  Replace caller-supplied provenance paths so a reused
  # output directory cannot select artifacts from another invocation.
  RESIDENT_ENV_JSON="/out/$(basename "${ENV_JSON}")"
  RESIDENT_POWER_TRACE="$(basename "${POWER_CSV}")"
  RESIDENT_ARGS=()
  HAS_POWER_TRACE_ARG=0
  RESIDENT_ARGUMENT_INDEX=0
  while [[ "${RESIDENT_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${RESIDENT_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --env-json)
        if [[ "${RESIDENT_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--env-json requires a value" >&2
          exit 2
        fi
        RESIDENT_ARGUMENT_INDEX=$((RESIDENT_ARGUMENT_INDEX + 2))
        continue
        ;;
      --env-json=*)
        RESIDENT_ARGUMENT_INDEX=$((RESIDENT_ARGUMENT_INDEX + 1))
        continue
        ;;
      --power-trace)
        if [[ "${RESIDENT_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--power-trace requires a value" >&2
          exit 2
        fi
        HAS_POWER_TRACE_ARG=1
        if [[ "${TELEMETRY_MODE}" != "off" ]]; then
          RESIDENT_ARGS+=("--power-trace" "${RESIDENT_POWER_TRACE}")
        fi
        RESIDENT_ARGUMENT_INDEX=$((RESIDENT_ARGUMENT_INDEX + 2))
        continue
        ;;
      --power-trace=*)
        HAS_POWER_TRACE_ARG=1
        if [[ "${TELEMETRY_MODE}" != "off" ]]; then
          RESIDENT_ARGS+=("--power-trace=${RESIDENT_POWER_TRACE}")
        fi
        RESIDENT_ARGUMENT_INDEX=$((RESIDENT_ARGUMENT_INDEX + 1))
        continue
        ;;
      *)
        RESIDENT_ARGS+=("${argument}")
        RESIDENT_ARGUMENT_INDEX=$((RESIDENT_ARGUMENT_INDEX + 1))
        ;;
    esac
  done
  RESIDENT_ARGS+=(--env-json "${RESIDENT_ENV_JSON}")
  if [[ "${TELEMETRY_MODE}" != "off" && "${HAS_POWER_TRACE_ARG}" == "0" ]]; then
    RESIDENT_ARGS+=(--power-trace "${RESIDENT_POWER_TRACE}")
  fi
  RUNNER_ARGS=("${RESIDENT_ARGS[@]}")
fi
if [[ "${RUNNER}" == "${ISSUE137_RUNNER_RELATIVE_PATH}" ]]; then
  # Issue #137 has one wrapper-owned result path.  Do not forward a caller
  # --out, and bind environment/trace names to this fresh wrapper identity.
  ISSUE137_ARGS=()
  HAS_POWER_TRACE_ARG=0
  ISSUE137_ARGUMENT_INDEX=0
  while [[ "${ISSUE137_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${ISSUE137_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --out)
        if [[ "${ISSUE137_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--out requires a value" >&2
          exit 2
        fi
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 2))
        continue
        ;;
      --out=*)
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 1))
        continue
        ;;
      --env-json)
        if [[ "${ISSUE137_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--env-json requires a value" >&2
          exit 2
        fi
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 2))
        continue
        ;;
      --env-json=*)
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 1))
        continue
        ;;
      --power-trace)
        if [[ "${ISSUE137_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--power-trace requires a value" >&2
          exit 2
        fi
        if [[ "${HAS_POWER_TRACE_ARG}" == "0" && "${TELEMETRY_MODE}" != "off" ]]; then
          ISSUE137_ARGS+=(--power-trace "/out/$(basename "${POWER_CSV}")")
        fi
        HAS_POWER_TRACE_ARG=1
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 2))
        continue
        ;;
      --power-trace=*)
        if [[ "${HAS_POWER_TRACE_ARG}" == "0" && "${TELEMETRY_MODE}" != "off" ]]; then
          ISSUE137_ARGS+=(--power-trace="/out/$(basename "${POWER_CSV}")")
        fi
        HAS_POWER_TRACE_ARG=1
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 1))
        continue
        ;;
      *)
        ISSUE137_ARGS+=("${argument}")
        ISSUE137_ARGUMENT_INDEX=$((ISSUE137_ARGUMENT_INDEX + 1))
        ;;
    esac
  done
  ISSUE137_ARGS+=(--env-json "/out/$(basename "${ENV_JSON}")")
  if [[ "${TELEMETRY_MODE}" != "off" && "${HAS_POWER_TRACE_ARG}" == "0" ]]; then
    ISSUE137_ARGS+=(--power-trace "/out/$(basename "${POWER_CSV}")")
  fi
  RUNNER_ARGS=("${ISSUE137_ARGS[@]}")
fi
if [[ "${RUNNER}" == "${ISSUE127_RUNNER_RELATIVE_PATH}" ]]; then
  # The generic custom-runner branch remains the execution path, but pass the
  # wrapper-owned sibling artifacts explicitly so run binding is visible in
  # the runner CLI and board-free tests can verify the seam.
  ISSUE127_ARGS=()
  HAS_ENV_JSON_ARG=0
  HAS_POWER_TRACE_ARG=0
  ISSUE127_ARGUMENT_INDEX=0
  while [[ "${ISSUE127_ARGUMENT_INDEX}" -lt "${#RUNNER_ARGS[@]}" ]]; do
    argument="${RUNNER_ARGS[${ISSUE127_ARGUMENT_INDEX}]}"
    case "${argument}" in
      --env-json)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--env-json requires a value" >&2
          exit 2
        fi
        HAS_ENV_JSON_ARG=1
        ISSUE127_ARGS+=("--env-json" "/out/$(basename "${ENV_JSON}")")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --env-json=*)
        HAS_ENV_JSON_ARG=1
        ISSUE127_ARGS+=("--env-json=/out/$(basename "${ENV_JSON}")")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
        continue
        ;;
      --power-trace)
        if [[ "${ISSUE127_ARGUMENT_INDEX}" -eq "$(( ${#RUNNER_ARGS[@]} - 1 ))" ]]; then
          echo "--power-trace requires a value" >&2
          exit 2
        fi
        HAS_POWER_TRACE_ARG=1
        ISSUE127_ARGS+=("--power-trace" "/out/$(basename "${POWER_CSV}")")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 2))
        continue
        ;;
      --power-trace=*)
        HAS_POWER_TRACE_ARG=1
        ISSUE127_ARGS+=("--power-trace=/out/$(basename "${POWER_CSV}")")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
        continue
        ;;
      *)
        ISSUE127_ARGS+=("${argument}")
        ISSUE127_ARGUMENT_INDEX=$((ISSUE127_ARGUMENT_INDEX + 1))
        ;;
    esac
  done
  if [[ "${HAS_ENV_JSON_ARG}" == "0" ]]; then
    ISSUE127_ARGS+=(--env-json "/out/$(basename "${ENV_JSON}")")
  fi
  if [[ "${TELEMETRY_MODE}" != "off" && "${HAS_POWER_TRACE_ARG}" == "0" ]]; then
    ISSUE127_ARGS+=(--power-trace "/out/$(basename "${POWER_CSV}")")
  fi
  RUNNER_ARGS=("${ISSUE127_ARGS[@]}")
fi
# `timeout` is inside the wrapper, so losing an SSH session cannot leave the
# Docker client or the board-side container unbounded.
if [[ "${TEST_MODE}" == "1" ]]; then
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v "${HUGEPAGES_SOURCE}:/dev/hugepages-1G" \
    -v "${REPO_ROOT_SOURCE}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    -e "HEKATUS_TT_RUN_ID=${RUN_ID}" \
    "${TELEMETRY_ENV[@]}" \
    -e HEKATUS_TT_DEVICE_TEST=1 \
    -e HEKATUS_TT_PINNED_CONTAINER=1 \
    "${WATCHER_ENV[@]}" \
    --entrypoint /usr/local/bin/uv \
    "${IMAGE}" run --no-project --with pytest==8.3.5 --with scipy==1.13.1 python -m pytest "${PYTEST_ARGS[@]}" &
elif [[ "${RUNNER}" == "${MATMUL_RUNNER_RELATIVE_PATH}" ]]; then
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v "${HUGEPAGES_SOURCE}:/dev/hugepages-1G" \
    -v "${REPO_ROOT_SOURCE}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    -e "HEKATUS_TT_RUN_ID=${RUN_ID}" \
    "${WATCHER_ENV[@]}" \
    "${TELEMETRY_ENV[@]}" \
    "${RUNNER_RESULT_ENV[@]}" \
    --entrypoint /bin/bash \
    "${IMAGE}" -lc 'exec python3 "$0" --out "$1" --env-json "$2" "${@:3}"' \
    "${RUNNER}" \
    "${RESULTS_CONTAINER}" \
    "/out/$(basename "${ENV_JSON}")" \
    "${RUNNER_ARGS[@]}" &
else
  timeout --signal=TERM --kill-after=5s "${CONTAINER_TIMEOUT_S}s" \
    docker run --rm --name "${CONTAINER_NAME}" \
    --device "${DEVICE_NODE}" \
    -v "${HUGEPAGES_SOURCE}:/dev/hugepages-1G" \
    -v "${REPO_ROOT_SOURCE}:/work" \
    -v "${OUT_DIR}:/out" \
    -w /work \
    -e PYTHONPATH=/work \
    -e "HEKATUS_TT_RUN_ID=${RUN_ID}" \
    "${WATCHER_ENV[@]}" \
    "${TELEMETRY_ENV[@]}" \
    "${RUNNER_RESULT_ENV[@]}" \
    --entrypoint python3 \
    "${IMAGE}" "${RUNNER}" "${RUNNER_ARGS[@]}" &
fi
DOCKER_PID=$!

# A sampler that exits before Docker finishes means the run has no complete
# power trace. Wait for whichever process finishes first so that this failure
# cannot be hidden by the normal shutdown signal sent after Docker completes.
# Each status is captured on the line after its own `wait`: any command in
# between — an assignment included — replaces $? with its own success.
set +e
if [[ "${TELEMETRY_MODE}" == "off" ]]; then
  wait "${DOCKER_PID}"
  DOCKER_STATUS=$?
  SAMPLER_STATUS=intentional
else
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
    kill -USR1 "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}"
  fi
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
echo "results     -> ${RESULT_PATH}"
if [[ "${TELEMETRY_MODE}" == "off" ]]; then
  echo "telemetry sampler -> off (diagnostic-only; no power trace by design)"
else
  echo "telemetry sampler -> ${TELEMETRY_MODE} (${TELEMETRY_INTERVAL_S}s)"
  echo "power trace -> ${POWER_CSV}"
fi
