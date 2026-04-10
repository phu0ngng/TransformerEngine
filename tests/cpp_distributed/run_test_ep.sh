#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Launch the EP distributed unit test across multiple GPUs.
#
# Usage:
#   bash run_test_ep.sh [num_gpus] [path/to/test_ep binary]
#
# Defaults:
#   num_gpus  = number of GPUs visible to nvidia-smi
#   test_ep   = ./build/test_ep  (relative to this script's directory)
#
# The script spawns one process per GPU, each with --process-id=<i>
# and --num-processes=<N>.  ncclUniqueId exchange happens via a temp
# file — no MPI needed.
#
# Environment variables:
#   GTEST_FILTER   — forwarded to all processes (e.g., "EPPipelineTest.*")
#   TE_EP_UID_FILE — override the shared uid file path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
TEST_BIN="${2:-${SCRIPT_DIR}/build/test_ep}"

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "ERROR: test binary not found or not executable: ${TEST_BIN}"
  echo "Build it first:  cd ${SCRIPT_DIR} && mkdir -p build && cd build && cmake .. -DNVTE_WITH_NCCL_EP=ON && make test_ep"
  exit 1
fi

if (( NUM_GPUS < 2 )); then
  echo "EP tests require at least 2 GPUs, found ${NUM_GPUS}. Skipping."
  exit 0
fi

# Unique temp file for this invocation (PID-stamped to avoid collisions)
UID_FILE="${TE_EP_UID_FILE:-/tmp/te_ep_test_uid_$$}"

echo "=== EP Distributed Test ==="
echo "  GPUs:     ${NUM_GPUS}"
echo "  Binary:   ${TEST_BIN}"
echo "  UID file: ${UID_FILE}"
echo

# Cleanup on exit
cleanup() {
  rm -f "${UID_FILE}"
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# Remove stale uid file if it exists
rm -f "${UID_FILE}"

PIDS=()
LOG_DIR=$(mktemp -d)

# Optional GTest filter
GTEST_ARGS=""
if [[ -n "${GTEST_FILTER:-}" ]]; then
  GTEST_ARGS="--gtest_filter=${GTEST_FILTER}"
fi

for i in $(seq 0 $((NUM_GPUS - 1))); do
  LOG_FILE="${LOG_DIR}/process_${i}.log"

  # Each process calls cudaSetDevice(process_id % device_count) internally.
  # Do NOT override CUDA_VISIBLE_DEVICES — this preserves SLURM GPU
  # mappings and allows NVLink topology to remain visible to NCCL.
  "${TEST_BIN}" \
    --process-id="${i}" \
    --num-processes="${NUM_GPUS}" \
    --uid-file="${UID_FILE}" \
    ${GTEST_ARGS} \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
done

# Wait for all processes and collect exit codes
HAS_FAILURE=0
for i in $(seq 0 $((NUM_GPUS - 1))); do
  if ! wait "${PIDS[$i]}"; then
    echo "FAILED: process ${i} (PID ${PIDS[$i]})"
    HAS_FAILURE=1
  fi
done

# Print process 0's output (the one with summary messages)
echo
echo "=== Process 0 output ==="
cat "${LOG_DIR}/process_0.log"

# On failure, also print other processes' logs
if (( HAS_FAILURE )); then
  for i in $(seq 1 $((NUM_GPUS - 1))); do
    echo
    echo "=== Process ${i} output ==="
    cat "${LOG_DIR}/process_${i}.log"
  done
  echo
  echo "=== SOME PROCESSES FAILED ==="
else
  echo
  echo "=== ALL PROCESSES PASSED ==="
fi

# Cleanup temp logs
rm -rf "${LOG_DIR}"

exit "${HAS_FAILURE}"
