# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

#!/bin/bash

SCRIPT_NAMES="${SCRIPT_NAMES:-test_multi_process_ep.py}"
TEST_TIMEOUT_S="${TEST_TIMEOUT_S:-180}"


XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_command_buffer=''"

export XLA_FLAGS="${XLA_BASE_FLAGS}"

# Ensure the in-tree TE source wins over a stale system-installed copy.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${TE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_RUNS=$(nvidia-smi -L | wc -l)

OVERALL_RET=0

for SCRIPT_NAME in $SCRIPT_NAMES; do
  echo "=== Running ${SCRIPT_NAME} ==="
  # NCCL EP needs all GPUs visible per process (peer-access enable loop) — pin via JAX local_device_ids instead of CUDA_VISIBLE_DEVICES.
  # Capture per-rank logs so non-rank-0 crashes are visible.
  for ((i=1; i<NUM_RUNS; i++))
  do
      timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
          python $SCRIPT_NAME 127.0.0.1:12345 $i $NUM_RUNS > stdout_rank_${i}.txt 2>&1 &
  done

  timeout --foreground --signal=KILL "${TEST_TIMEOUT_S}" \
      python $SCRIPT_NAME 127.0.0.1:12345 0 $NUM_RUNS 2>&1 | tee stdout_multi_process.txt

  wait

  RET=0
  if grep -q "FAILED" stdout_multi_process.txt; then
    RET=1
  fi
  # If rank 0 produced no "OK" / "PASSED" / "Ran N tests" line, treat as hang/crash
  # rather than silent success. Without this guard, a SIGKILL'd rank exits with no
  # output and the launcher would exit 0.
  if ! grep -qE "Ran [0-9]+ test|^OK$|PASSED" stdout_multi_process.txt; then
    echo "ERROR: rank 0 produced no test summary for ${SCRIPT_NAME} — likely a hang or early crash."
    echo "      (See stdout_multi_process.txt; was the EP communicator able to init?"
    echo "       NCCL EP requires NVLS multicast — check 'NVLS multicast support'"
    echo "       in NCCL_DEBUG=INFO output.)"
    RET=1
  fi
  if [ "$RET" -ne 0 ]; then
    for ((i=1; i<NUM_RUNS; i++)); do
      echo "--- rank $i log ---"
      cat stdout_rank_${i}.txt 2>/dev/null || echo "(no log)"
    done
  fi

  rm -f stdout_multi_process.txt stdout_rank_*.txt
  if [ "$RET" -ne 0 ]; then
    OVERALL_RET=1
  fi
done

exit "$OVERALL_RET"
