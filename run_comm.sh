RUN_NAME=${1:-"testing_te_cm"}
DP=${2:-1}
FSDP=${3:-2}
TP=${4:-4}
BATCH_SIZE=${5:-8}
NUM_GPUS=${6:-8}
COMM=${7:-"AG"}
SQ=${8:-4096}
HID_SIZE=${9:-4096}
FFN_SIZE=${10:-14336}


export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

#export CUDA_LAUNCH_BLOCKING=1
AR_THRESHOLD=302212254
AG_THRESHOLD=302212254
RS_THRESHOLD=50331648

export CUDA_DEVICE_MAX_CONNECTIONS=16

mkdir -p profiles/nsys_logs/${RUN_NAME}
NSYS_OUTPUT_FILE=profiles/nsys_logs/${RUN_NAME}/${RUN_NAME}
HLO_PATH=profiles/nsys_logs/${RUN_NAME}/${RUN_NAME}-hlo

echo "RUNNING IN PERFORMANCE MODE"

export XLA_FLAGS="
    --xla_gpu_enable_latency_hiding_scheduler=true
    --xla_gpu_enable_triton_gemm=false
    --xla_gpu_enable_host_memory_offloading=false
    --xla_gpu_enable_command_buffer=""
    --xla_gpu_all_reduce_combine_threshold_bytes=${AR_THRESHOLD}
    --xla_gpu_all_gather_combine_threshold_bytes=${AG_THRESHOLD}
    --xla_gpu_reduce_scatter_combine_threshold_bytes=${RS_THRESHOLD}
    --xla_gpu_enable_pipelined_all_gather=false
    --xla_gpu_enable_pipelined_reduce_scatter=false
    --xla_gpu_enable_pipelined_all_reduce=false
    --xla_gpu_enable_while_loop_double_buffering=true
    --xla_gpu_enable_all_gather_combine_by_dim=false
    --xla_gpu_enable_reduce_scatter_combine_by_dim=false
    --xla_disable_hlo_passes=rematerialization
    --xla_gpu_enable_custom_fusions=false
    --xla_dump_hlo_as_text
    --xla_dump_hlo_as_html
    --xla_dump_to=${HLO_PATH}"

NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE}-perf --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
echo ${XLA_FLAGS}
echo "--dp-size $DP --tp-size $TP --fsdp-size $FSDP --batch-size $BATCH_SIZE --seq-length $SQ --hidden-size $HID_SIZE --activation-size $FFN_SIZE --comm-type $COMM --fp8"

export JAX_USE_SHARDY_PARTITIONER=false

# mpirun --oversubscribe -n ${NUM_GPUS} --allow-run-as-root ${NSYS_CMD} python examples/jax/comm_overlap/layer_prim_with_overlap.py --dp-size $DP --tp-size $TP --fsdp-size $FSDP --batch-size $BATCH_SIZE --seq-length $SQ --hidden-size $HID_SIZE --activation-size $FFN_SIZE
mpirun -n ${NUM_GPUS} ${NSYS_CMD} python examples/jax/comm_overlap/layer_prim_with_overlap.py --dp-size $DP --tp-size $TP --fsdp-size $FSDP --batch-size $BATCH_SIZE --seq-length $SQ --hidden-size $HID_SIZE --activation-size $FFN_SIZE
