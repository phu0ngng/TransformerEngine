#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Run TE EP distributed unit tests across multiple GPUs.
#
# Spawns one background bash process per GPU (no MPI dependency), matching the
# JAX multi-process launcher style. ncclUniqueId is exchanged via a shared
# temp file (see test_ep_common.h).
#
# Each test binary is run twice — once per init path:
#   uid path  : nvte_ep_initialize        (JAX / generic)
#   comm path : nvte_ep_initialize_with_comm  (PyTorch)
#
# Usage:
#   bash run_test_ep.sh [num_gpus] [build_dir]
#
# Defaults:
#   num_gpus  = number of GPUs visible to nvidia-smi
#   build_dir = <script_dir>/build
#
# Environment variables:
#   GTEST_FILTER     — forwarded to all processes (e.g., "EPDispatchTest.*")
#   TEST_TIMEOUT_S   — per-process timeout in seconds (default: 180)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${2:-${SCRIPT_DIR}/build}"
NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-180}"

GTEST_ARGS="${GTEST_FILTER:+--gtest_filter=${GTEST_FILTER}}"
OVERALL_FAIL=0

# ---------------------------------------------------------------------------
# run_suite BINARY SUITE_NAME MIN_GPUS [--use-comm]
# ---------------------------------------------------------------------------
run_suite() {
    local BINARY="$1"
    local SUITE_NAME="$2"
    local MIN_GPUS="${3:-2}"
    local USE_COMM="${4:-}"          # "--use-comm" or empty

    local TEST_BIN="${BUILD_DIR}/${BINARY}"

    if [[ ! -x "${TEST_BIN}" ]]; then
        echo "ERROR: binary not found: ${TEST_BIN}"
        echo "Build:  cd ${SCRIPT_DIR} && mkdir -p build && cd build && cmake .. -DNVTE_WITH_NCCL_EP=ON && make"
        OVERALL_FAIL=1
        return
    fi

    if (( NUM_GPUS < MIN_GPUS )); then
        echo "${SUITE_NAME}: requires ${MIN_GPUS} GPUs, found ${NUM_GPUS}. Skipping."
        return
    fi

    local TMPDIR_L="${TMPDIR:-/tmp}"
    local UID_FILE="${TMPDIR_L}/te_ep_uid_${BINARY}_${USE_COMM:+comm_}$$"
    rm -f "${UID_FILE}"

    local LOG_DIR
    LOG_DIR=$(mktemp -d)
    local FAIL=0

    echo "=== ${SUITE_NAME} ==="
    echo "  GPUs: ${NUM_GPUS}   Binary: ${TEST_BIN}"
    echo

    # Spawn one background process per GPU. ncclUniqueId is exchanged via the
    # shared UID_FILE. Each process is wrapped in `timeout` to detect hangs early.
    local PIDS=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
            "${TEST_BIN}" \
                --rank="${i}" \
                --nranks="${NUM_GPUS}" \
                --uid-file="${UID_FILE}" \
                ${USE_COMM} \
                ${GTEST_ARGS} \
                > "${LOG_DIR}/rank_${i}.log" 2>&1 &
        PIDS+=($!)
    done
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        if ! wait "${PIDS[$i]}"; then
            local rc=$?
            FAIL=1
            if [[ $rc -eq 137 || $rc -eq 124 ]]; then
                echo "  rank ${i}: TIMEOUT after ${TEST_TIMEOUT_S}s (rc=${rc})"
            fi
        fi
    done

    echo "--- Rank 0 output ---"
    cat "${LOG_DIR}/rank_0.log"

    if (( FAIL )); then
        for i in $(seq 1 $((NUM_GPUS - 1))); do
            echo "--- Rank ${i} output ---"
            cat "${LOG_DIR}/rank_${i}.log"
        done
        echo "=== ${SUITE_NAME}: FAILED ==="
        OVERALL_FAIL=1
    else
        echo "=== ${SUITE_NAME}: ALL PASSED ==="
    fi

    rm -rf "${LOG_DIR}"
    rm -f "${UID_FILE}"
}

# ---------------------------------------------------------------------------
# Cleanup on abort
# ---------------------------------------------------------------------------
cleanup() { rm -f "${TMPDIR:-/tmp}"/te_ep_uid_*_"$$" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Run all suites for both init paths
# ---------------------------------------------------------------------------
run_suite "test_ep_init"     "EP Init Tests (uid)"              2
run_suite "test_ep_init"     "EP Init Tests (comm)"             2 "--use-comm"

echo

run_suite "test_ep_pipeline" "EP Pipeline Tests (uid)"          2
run_suite "test_ep_pipeline" "EP Pipeline Tests (comm)"         2 "--use-comm"

echo
if (( OVERALL_FAIL )); then
    echo "=== SOME SUITES FAILED ==="
else
    echo "=== ALL SUITES PASSED ==="
fi

exit "${OVERALL_FAIL}"
