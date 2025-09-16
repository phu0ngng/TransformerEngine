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

    # Run pytest and redirect stdout and stderr to the log file
    pytest -s -c "$TE_PATH/tests/jax/pytest.ini" \
      -vs "$TE_PATH/examples/jax/collective_gemm/$TEST_FILE" \
      --num-process=$NUM_GPUS \
      --process-id=$i  > "$LOG_FILE" 2>&1 &
    done

  # Wait for the processes to finish
  wait
  # tail -n +7 "${TEST_FILE}_gpu_0.log"
  cat "${TEST_FILE}_gpu_0.log"

  # Check and print the log content accordingly
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
