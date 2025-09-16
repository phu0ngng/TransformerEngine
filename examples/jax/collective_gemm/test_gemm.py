# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Collective GEMM test on multi-GPU with tensor parallelism

This script uses custom distributed initialization with the following arguments:
- --coordinator-address: Coordinator address for distributed initialization
- --num-processes: Number of processes for distributed initialization
- --process-id: Process ID for distributed initialization
- --local-device-ids: Local device IDs for distributed initialization

Example:
    python test_gemm.py --coordinator-address localhost:1234 --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3
"""
import argparse
import unittest
import os

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding

from common import assert_allclose, assert_allclose_print_index

import transformer_engine.jax.cpp_extensions as tex

# from transformer_engine.jax.quantize import is_fp8_available, ScalingMode, Quantizer, QuantizeConfig, fp8_autocast
from transformer_engine.jax.quantize import fp8_autocast
from transformer_engine.jax.cpp_extensions.gemm import (
    CollectiveGemmConfig,
    CollectiveOp,
    noop_cgemm_config,
)
from transformer_engine.jax.sharding import MeshResource

DEVICE_DP_AXIS = "data"
DEVICE_TPSP_AXIS = "tensor_sequence"
PARAMS_KEY = "params"

jax.clear_caches()
jax.config.update(
    "jax_use_shardy_partitioner", False
)  # CollectiveGEMM does not work with Shardy yet

# Global flag to track if distributed has been initialized
_distributed_initialized = False

def _is_distributed_initialized():
    """Check if JAX distributed has been initialized."""
    return _distributed_initialized

def _initialize_distributed(args):
    """Initialize JAX distributed with custom arguments."""
    global _distributed_initialized

    # Check if already initialized
    if _distributed_initialized:
        print("JAX distributed already initialized, skipping...")
        return

    if args.coordinator_address is None or args.num_processes is None or args.process_id is None:
        raise ValueError(
            "All distributed initialization arguments are required: "
            "--coordinator-address, --num-processes, --process-id"
        )
    if args.local_device_ids is None:
        assert args.num_devices_per_process is not None, "Either local_device_ids or num_devices_per_process must be provided"
        # Calculate device range for this process
        # Single process single device: each process gets one unique device
        # Single process multiple devices: each process gets a unique range of devices
        start_device = args.process_id * args.num_devices_per_process
        device_range = range(start_device, start_device + args.num_devices_per_process)
        global_device_ids_for_this_process = ",".join(map(str, device_range))
    else:
        # Use explicitly provided global device IDs
        global_device_ids_for_this_process = args.local_device_ids
        args.num_devices_per_process = len(args.local_device_ids.split(","))

    assert args.num_devices_per_process == 1, "Only single process single GPU is supported!"

    print(f"Initializing JAX distributed with coordinator={args.coordinator_address}, "
          f"num_processes={args.num_processes}, process_id={args.process_id}")
    print(f"This process will manage global CUDA devices: {global_device_ids_for_this_process}")
    
    # Validate device assignment
    device_list = global_device_ids_for_this_process.split(",")
    print(f"Process {args.process_id} assigned devices: {device_list}")
    
    if args.num_devices_per_process == 1:
        # Single device per process: validate device = process_id
        assert len(device_list) == 1, f"Expected 1 device per process, got {len(device_list)} devices: {device_list}"
        expected_device = str(args.process_id)
        actual_device = device_list[0]
        print(f"Single device per process: process {args.process_id} → device {actual_device}")
        assert actual_device == expected_device, f"Device assignment mismatch: process {args.process_id} should use device {expected_device}, but got {actual_device}"
    else:
        # Multiple devices per process: validate device range
        expected_start = args.process_id * args.num_devices_per_process
        expected_devices = [str(i) for i in range(expected_start, expected_start + args.num_devices_per_process)]
        print(f"Multiple devices per process: process {args.process_id} → devices {device_list}")
        assert device_list == expected_devices, f"Device range mismatch: process {args.process_id} should use devices {expected_devices}, but got {device_list}"

    # Note: "local_device_ids" is a JAX term meaning "global CUDA devices managed by this process"
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=global_device_ids_for_this_process,
    )

    # Mark as initialized
    _distributed_initialized = True

    assert (
        jax.local_device_count() == 1
    ), f"[{args.process_id}|{args.num_devices_per_process}] Expected 1 GPU per process, found {jax.local_device_count()}"

    # Initialize CGEMM communicator for single device per process scenario
    # num_ranks = total ranks across all processes (args.num_processes in this case)
    # num_local_ranks = GPUs per process (1 for single device per process)
    num_local_ranks = 1  # Single GPU per process
    total_ranks = args.num_processes  # Total number of processes/ranks
    
    print(f"Initializing CGEMM communicator with num_ranks={total_ranks}, num_local_ranks={num_local_ranks}, process_id={args.process_id}")
    print(f"JAX local devices: {jax.local_devices()}")
    print(f"JAX device count: {jax.local_device_count()}")
    
    tex.initialize_cgemm_communicator(num_ranks=total_ranks, num_local_ranks=num_local_ranks, process_id=args.process_id)


def _get_operand_sharding(mesh, collective_op, is_with_dp):

    dp_axis = DEVICE_DP_AXIS if is_with_dp else None
    if collective_op == CollectiveOp.ALL_GATHER:
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, DEVICE_TPSP_AXIS, None))
        weight_sharding = NamedSharding(mesh, PartitionSpec(None, DEVICE_TPSP_AXIS))
        bias_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_TPSP_AXIS))
    else:  # RS
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, DEVICE_TPSP_AXIS))
        weight_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_TPSP_AXIS, None))
        bias_sharding = NamedSharding(mesh, PartitionSpec(None))

    return x_sharding, weight_sharding, bias_sharding


def _create_mesh(args):
    """Create mesh configuration with proper validation."""
    num_gpu = jax.device_count()
    numranks = args.num_processes * args.num_devices_per_process
    assert num_gpu == numranks, f"Requires {num_gpu} processes for {num_gpu} GPUs, got {numranks}!"
    num_gpu_dp = 2 if args.enable_data_parallel else 1
    assert (
        num_gpu > 1 and num_gpu % num_gpu_dp == 0
    ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"

    num_gpu_tp = num_gpu // num_gpu_dp
    assert num_gpu_tp > 1, f"Number of GPUs for tensor parallelism ({num_gpu_tp}) must be > 1"
    print(f"Using {num_gpu_dp}x{num_gpu_tp} mesh ({num_gpu_dp * num_gpu_tp} total GPUs)")

    device_mesh = mesh_utils.create_device_mesh((num_gpu_dp, num_gpu_tp))
    mesh = jax.sharding.Mesh(devices=device_mesh, axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS))
    jax.sharding.set_mesh(mesh)

    return mesh


def _jitted_cgemm(x, weight, bias, contracting_dims, cgemm_config):
    return jax.jit(tex.gemm, static_argnames=("contracting_dims", "cgemm_config"))(
        x,
        weight,
        bias=bias,
        contracting_dims=contracting_dims,
        cgemm_config=cgemm_config,
    )


def run_gemm_tests(args, mesh=None):
    """Execute GEMM tests."""
    print(args)
    # Collective GEMM requires Shardy partitioner to be disabled
    jax.config.update("jax_use_shardy_partitioner", False)

    # Initialize distributed with provided arguments
    _initialize_distributed(args)

    n_gpus = args.num_devices_per_process * args.num_processes
    mesh = mesh or _create_mesh(args)

    # Create test data
    rng = jax.random.PRNGKey(0)
    rng, x_rng, weight_rng, bias_rng = jax.random.split(rng, 4)
    x = jax.random.normal(
        x_rng, (args.batch_size, args.seq_len, args.hidden_in), dtype=jnp.bfloat16
    )
    weight = jax.random.normal(weight_rng, (args.hidden_in, args.hidden_out), dtype=jnp.bfloat16)
    bias = jax.random.normal(bias_rng, (args.hidden_out,), dtype=jnp.bfloat16)

    with mesh, fp8_autocast(
        enabled=False,
        fp8_recipe=None,
        mesh_resource=MeshResource(dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS),
    ):
        print(f"Device mesh: {mesh}")

        # Collective GEMM configs need to be created under the mesh_resource context
        collective_op = (
            CollectiveOp.ALL_GATHER
            if args.collective_type == "all_gather"
            else CollectiveOp.REDUCE_SCATTER
        )
        cgemm_config = CollectiveGemmConfig.create(collective_op=collective_op)

        x_sharding, weight_sharding, bias_sharding = _get_operand_sharding(
            mesh, collective_op, args.enable_data_parallel
        )
        x_sharded = jax.device_put(x, x_sharding)
        weight_sharded = jax.device_put(weight, weight_sharding)
        bias_sharded = jax.device_put(bias, bias_sharding)

        ref_output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            contracting_dims=((2,), (0,)),
            cgemm_config=noop_cgemm_config,
        )
        output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            contracting_dims=((2,), (0,)),
            cgemm_config=cgemm_config,
        )
        gathered_ref_output = jax.lax.with_sharding_constraint(
            ref_output, NamedSharding(mesh, PartitionSpec(None))
        )
        gathered_output = jax.lax.with_sharding_constraint(
            output, NamedSharding(mesh, PartitionSpec(None))
        )
        jax.block_until_ready(gathered_ref_output)
        jax.block_until_ready(gathered_output)

    if args.enable_result_check and args.process_id == 0:
        assert_allclose(gathered_ref_output, gathered_output)
        # assert_allclose(gathered_ref_output, gathered_output, atol=1e-3, rtol=2e-2)
        # assert_allclose_print_index(gathered_ref_output, gathered_output)
        # assert_allclose_print_index(gathered_ref_output, gathered_output, rtol=1e-2, atol=1e-5)


def gemm_parser(args):
    """Test settings."""
    parser = argparse.ArgumentParser(description="JAX Collective GEMM Test")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size (default: 4)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        metavar="N",
        help="sequence length (default: 2048)",
    )
    parser.add_argument(
        "--hidden-in",
        type=int,
        default=1024,
        metavar="N",
        help="input hidden dimension (default: 1024)",
    )
    parser.add_argument(
        "--hidden-out",
        type=int,
        default=2048,
        metavar="N",
        help="output hidden dimension (default: 2048)",
    )
    parser.add_argument(
        "--collective-type",
        type=str,
        default="all_gather",
        choices=["all_gather", "reduce_scatter"],
        help="Collective operation type (default: all_gather)",
    )
    parser.add_argument(
        "--fp8-recipe",
        type=str,
        default="DelayedScaling",
        help="FP8 recipe (default: DelayedScaling)",
    )
    parser.add_argument(
        "--enable-data-parallel",
        action="store_true",
        default=False,
        help="Enable data parallel (default: False)",
    )
    parser.add_argument(
        "--enable-result-check",
        action="store_true",
        default=False,
        help="Enable result check (default: False)",
    )
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default="127.0.0.1:1234",
        help=(
            "the IP address of process 0 and a port on which that"
            " process should launch a coordinator service (default:"
            " 127.0.0.1:1234)"
        ),
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for distributed initialization",
    )
    parser.add_argument(
        "--num-devices-per-process",
        type=int,
        default=1,
        help="Number of devices per process for distributed initialization",
    )
    parser.add_argument(
        "--process-id",
        type=int,
        default=None,
        help="Process ID for distributed initialization",
    )
    parser.add_argument(
        "--local-device-ids",
        type=str,
        default=None,
        help="List of local device IDs for distributed initialization",
    )
    return parser.parse_args(args)


class TestCollectiveGemm(unittest.TestCase):
    """Collective GEMM unittests"""

    # is_fp8_supported, fp8_reason = is_fp8_available(ScalingMode.DELAYED_TENSOR_SCALING)
    # is_mxfp8_supported, mxfp8_reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

    def setUp(self):
        """Set up test environment for pytest execution."""
        # Create args object with distributed parameters from pytest fixtures
        self.args = gemm_parser([])
        self.args.coordinator_address = self.coordinator_address
        self.args.num_processes = self.num_processes
        self.args.process_id = self.process_id
        self.args.local_device_ids = self.local_device_ids
        self.args.num_devices_per_process = self.num_devices_per_process
        _initialize_distributed(self.args)
        # Create mesh once for all tests
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"
        self.args.batch_size = 1
        self.args.seq_len = 128
        self.args.hidden_in = 64
        self.args.hidden_out = 64

    def tearDown(self):
        """Clean up after each test."""
        # Clear the mesh to prevent interference between tests
        # Note: JAX doesn't accept None, so we create a minimal 1-device mesh to reset
        single_device_mesh = jax.sharding.Mesh([jax.devices()[0]], axis_names=())
        jax.sharding.set_mesh(single_device_mesh)
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather(self):
        """Test Collective GEMM with AllGather"""
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    # def test_te_bf16_reduce_scatter(self):
    #     """Test Collective GEMM with ReduceScatter"""
    #     self.args.collective_type = "reduce_scatter"
    #     run_gemm_tests(self.args, self.mesh)


# class TestCollectiveGemmWithDP(unittest.TestCase):
#     """Collective GEMM with DP unittests"""
#
#     def setUp(self):
#         """Set up test environment for pytest execution."""
#         # Create args object with distributed parameters from pytest fixtures
#         self.args = gemm_parser([])
#         self.args.coordinator_address = self.coordinator_address
#         self.args.num_processes = self.num_processes
#         self.args.process_id = self.process_id
#         self.args.local_device_ids = self.local_device_ids
#         self.args.num_devices_per_process = self.num_devices_per_process
#         _initialize_distributed(self.args)
#         # Create mesh once for all tests
#         self.args.enable_data_parallel = True
#         self.mesh = _create_mesh(self.args)
#         jax.sharding.set_mesh(self.mesh)
#         self.args.enable_result_check = True
#         os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"
#
#     def tearDown(self):
#         """Clean up after each test."""
#         # Clear the mesh to prevent interference between tests
#         jax.sharding.set_mesh(None)
#         os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)
#
#     def test_te_bf16_all_gather_with_dp(self):
#         """Test Collective GEMM with AllGather"""
#         self.args.collective_type = "all_gather"
#         run_gemm_tests(self.args, self.mesh)
#
#     def test_te_bf16_reduce_scatter_with_dp(self):
#         """Test Collective GEMM with ReduceScatter"""
#         self.args.collective_type = "reduce_scatter"
#         run_gemm_tests(self.args, self.mesh)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 7:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print("Usage: python test_gemm.py --coordinator-address <address> --num-processes <num> --process-id <id> [--local-device-ids <ids>] [other args]")
        print("Example: python test_gemm.py --coordinator-address localhost:1234 --num-processes 4 --process-id 0")
        print("Example: python test_gemm.py --coordinator-address localhost:1234 --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3")
        sys.exit(1)

    args = gemm_parser(None)
    _initialize_distributed(args)
    run_gemm_tests(args, mesh=None)
