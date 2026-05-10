# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#!/bin/bash

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

# NCCL EP requires NVLink P2P among ranks on the node.
echo "*** Checking NVLINK support ***"
NVLINK_OUTPUT=$(nvidia-smi nvlink --status 2>&1)
NVLINK_EXIT_CODE=$?
if [ $NVLINK_EXIT_CODE -ne 0 ] || [[ "$NVLINK_OUTPUT" == *"not supported"* ]] \
   || [[ "$NVLINK_OUTPUT" == *"No devices"* ]] || [ -z "$NVLINK_OUTPUT" ]; then
  echo "NVLINK is not supported on this platform — EP example requires NVLINK; SKIPPING"
  exit 0
fi
echo "NVLINK support detected"

TEST_CASES=(
"test_ep_pipeline.py::TestEPPipeline::test_moe_fwd"
"test_ep_pipeline.py::TestEPPipeline::test_moe_fwd_bwd"
)

echo
echo "*** Executing tests in examples/jax/ep/ ***"

HAS_FAILURE=0
PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for TEST_CASE in "${TEST_CASES[@]}"; do
  echo
  echo "=== Starting test: $TEST_CASE ..."
  TEST_NAME=$(echo "$TEST_CASE" | awk -F'::' '{print $NF}')
  PIDS=()

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    LOG_FILE="${TEST_NAME}_gpu_${i}.log"
    if [ $i -eq 0 ]; then
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs --junitxml=$XML_LOG_DIR/ep_${TEST_NAME}.xml \
        "$TE_PATH/examples/jax/ep/$TEST_CASE" \
        --num-processes=$NUM_GPUS --process-id=$i 2>&1 | tee "$LOG_FILE" &
      PIDS+=($!)
    else
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/ep/$TEST_CASE" \
        --num-processes=$NUM_GPUS --process-id=$i > "$LOG_FILE" 2>&1 &
      PIDS+=($!)
    fi
  done
  wait

  if grep -q "SKIPPED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE SKIPPED"
  elif grep -q "FAILED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE FAILED"
    HAS_FAILURE=1
  elif grep -q "PASSED" "${TEST_NAME}_gpu_0.log"; then
    echo "... $TEST_CASE PASSED"
  else
    echo "... $TEST_CASE INVALID"
    HAS_FAILURE=1
  fi
  rm ${TEST_NAME}_gpu_*.log
done

cleanup
exit $HAS_FAILURE
