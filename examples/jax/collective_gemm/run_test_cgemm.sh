# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# Define the test files to run
TEST_FILES=(
"test_gemm.py"
# "test_dense_grad.py"
# "test_layernorm_mlp_grad.py"
)

echo
echo "*** Executing tests in examples/jax/collective_gemm/ ***"

HAS_FAILURE=0  # Global failure flag

# Run each test file across all GPUs
for TEST_FILE in "${TEST_FILES[@]}"; do
  echo
  echo "=== Starting test file: $TEST_FILE ..."
  set +x

  for i in $(seq 0 $(($NUM_GPUS - 1))); do
    # Define output file for logs
    LOG_FILE="${TEST_FILE}_gpu_${i}.log"

    if [ $i -eq 0 ]; then
      # For process 0: show live output AND save to log file using tee
      echo "=== Starting process 0 with live output ==="
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_FILE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i 2>&1 | tee "$LOG_FILE" &
    else
      # For other processes: redirect to log files only
      pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
        -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_FILE" \
        --num-processes=$NUM_GPUS \
        --process-id=$i > "$LOG_FILE" 2>&1 &
    fi
  done

  # Wait for all processes to finish
  wait

  # Check and print the log content from process 0 (now has log file thanks to tee)
  if grep -q "SKIPPED" "${TEST_FILE}_gpu_0.log"; then
    echo "... $TEST_FILE SKIPPED"
  elif grep -q "PASSED" "${TEST_FILE}_gpu_0.log"; then
    echo "... $TEST_FILE PASSED"
  else
    HAS_FAILURE=1
    echo "... $TEST_FILE FAILED"
  fi

  # Remove the log files after processing them
  wait
  # rm ${TEST_FILE}_gpu_*.log
done

wait
exit $HAS_FAILURE