#!/bin/bash

if [ -z "$TAG" ]; then
  TAG="norms.txt"
fi

mkdir -p outputs
OUT="outputs/log_$TAG.txt"
REPORT="outputs/summary_$TAG.txt"
TMP="outputs/tmp.txt"

norms=("LAYERNORM" "RMSNORM")
directions=("FWD" "BWD")
zeros=("X0" "X1")

for norm in "${norms[@]}"; do
    for direction in "${directions[@]}"; do
        for zero in "${zeros[@]}"; do
            # Set the environment variable directly
            env_var="NVTE_${direction}_${norm}_USE_CUDNN=1"

            # Set the appropriate filter based on the norm type
            if [ "$norm" == "LAYERNORM" ]; then
                filter="*LN*"
            else
                filter="*RMS*"
            fi

            # Construct and execute the command
            cmd="$env_var ./build/operator/test_operator --gtest_filter=${filter}.*${zero}"
            echo "$cmd" >> "$OUT"
            eval "$cmd" > "$TMP"

            # Extract the number of failures and graphs
            num_failures=$(grep -oP '\d+(?=\s+FAILED TESTS)' "$TMP")
            num_graphs=$(grep -c "_graph.*" "$TMP")

            # Print the results
            if [ -z "$num_failures" ]; then
              num_failures=0
            fi
            if [ -z "$num_graphs" ]; then
              num_graphs=0
            fi
            printf "%s_%s_%s: %d total failures, %d graph failures\n" "$norm" "$direction" "$zero" "$num_failures" "$num_graphs" >> "$REPORT"
            cat $TMP >> $OUT
            rm $TMP
        done
    done
done
