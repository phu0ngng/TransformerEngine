# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Collective Dense Gradient test on multi-GPU with tensor parallelism"""
import argparse
import unittest
import os

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding
import flax

from common import assert_allclose

from transformer_engine.jax.dense import dense

# from transformer_engine.jax.quantize import is_fp8_available, ScalingMode, Quantizer, QuantizeConfig, fp8_autocast
from transformer_engine.jax.quantize import fp8_autocast
from transformer_engine.jax.cpp_extensions.gemm import (
    CollectiveGemmConfigSet,
    CollectiveOp,
    noop_cgemm_config_set,
    initialize_cgemm_communicator,
)
from transformer_engine.jax.sharding import MeshResource
import transformer_engine.jax.flax as te_flax

NAME_DP_AXIS = "data"
NAME_TPSP_AXIS = "tensor_sequence"
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
        args.local_device_ids = ",".join(map(str, range(args.num_devices_per_process)))
    else:
        args.num_devices_per_process = len(args.local_device_ids.split(","))

    assert args.num_devices_per_process == 1, "Only single process single GPU is supported!"

    print(f"Initializing JAX distributed with coordinator={args.coordinator_address}, "
          f"num_processes={args.num_processes}, process_id={args.process_id}, "
          f"local_device_ids={args.local_device_ids}")

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=args.local_device_ids,
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
    initialize_cgemm_communicator(num_ranks=total_ranks, num_local_ranks=num_local_ranks, process_id=args.process_id)


def _get_logical_axes(collective_op):
    if collective_op.is_all_gather:
        input_axes = (NAME_DP_AXIS, NAME_TPSP_AXIS, None)
        weight_axes = (None, NAME_TPSP_AXIS)
        bias_axes = (NAME_TPSP_AXIS,)
        output_axes = (NAME_DP_AXIS, None, NAME_TPSP_AXIS)
    else:  # RS
        input_axes = (NAME_DP_AXIS, None, NAME_TPSP_AXIS)
        weight_axes = (NAME_TPSP_AXIS, None)
        bias_axes = (None,)
        output_axes = (NAME_DP_AXIS, NAME_TPSP_AXIS, None)
    return input_axes, weight_axes, bias_axes, output_axes


def _get_operand_sharding(mesh, collective_op):
    input_axes, weight_axes, bias_axes, _ = _get_logical_axes(collective_op)
    x_sharding = NamedSharding(mesh, PartitionSpec(*input_axes))
    weight_sharding = NamedSharding(mesh, PartitionSpec(*weight_axes))
    bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_axes))
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
    mesh = jax.sharding.Mesh(devices=device_mesh, axis_names=(NAME_DP_AXIS, NAME_TPSP_AXIS))
    jax.sharding.set_mesh(mesh)

    return mesh


def _mean_dense(x, weight, bias, input_axes, weight_axes, output_axes, cgemm_config_set):
    output = dense(
        x,
        weight,
        bias,
        contracting_dims=((2,), (0,)),
        input_axes=input_axes,
        kernel_axes=weight_axes,
        output_axes=output_axes,
        cgemm_config_set=cgemm_config_set,
    )
    return jnp.mean(output.astype(jnp.float32))


def _value_and_grad_dense(x, weight, bias, input_axes, weight_axes, output_axes, cgemm_config_set):
    return jax.jit(jax.value_and_grad(_mean_dense, (0, 1, 2)), static_argnums=(3, 4, 5, 6))(
        x, weight, bias, input_axes, weight_axes, output_axes, cgemm_config_set
    )


def run_dense_grad_tests(args, mesh=None):
    """Execute Dense Gradient tests."""
    print(args)
    # Collective GEMM requires Shardy partitioner to be disabled
    jax.config.update("jax_use_shardy_partitioner", False)

    # Initialize distributed with provided arguments
    _initialize_distributed(args)

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
        mesh_resource=MeshResource(dp_resource=NAME_DP_AXIS, tpsp_resource=NAME_TPSP_AXIS),
    ):
        # Get the base axis rules and extend them with TE's rules. This must be done inside fp8_autocast
        axis_rules = flax.linen.get_logical_axis_rules()
        axis_rules += ((NAME_TPSP_AXIS, NAME_TPSP_AXIS), (NAME_DP_AXIS, NAME_DP_AXIS))
        te_extended_axis_rules = te_flax.extend_logical_axis_rules(axis_rules)
        with flax.linen.logical_axis_rules(te_extended_axis_rules):
            # Collective GEMM configs need to be created under the mesh_resource context
            collective_op = (
                CollectiveOp.ALL_GATHER
                if args.collective_type == "all_gather"
                else CollectiveOp.REDUCE_SCATTER
            )
            cgemm_config_set = CollectiveGemmConfigSet.create(forward_collective_op=collective_op)

            x_sharding, weight_sharding, bias_sharding = _get_operand_sharding(mesh, collective_op)
            x_sharded = jax.device_put(x, x_sharding)
            weight_sharded = jax.device_put(weight, weight_sharding)
            bias_sharded = jax.device_put(bias, bias_sharding)

            input_axes, weight_axes, _, output_axes = _get_logical_axes(collective_op)
            ref_output, ref_grads = _value_and_grad_dense(
                x_sharded,
                weight_sharded,
                bias_sharded,
                input_axes,
                weight_axes,
                output_axes,
                noop_cgemm_config_set,
            )
            output, sharded_grads = _value_and_grad_dense(
                x_sharded,
                weight_sharded,
                bias_sharded,
                input_axes,
                weight_axes,
                output_axes,
                cgemm_config_set,
            )
        jax.block_until_ready(ref_output)
        jax.block_until_ready(output)
        gathered_grads = []
        gathered_ref_grads = []
        for ref_grad, grad in zip(ref_grads, sharded_grads):
            gathered_grads.append(
                jax.lax.with_sharding_constraint(grad, NamedSharding(mesh, PartitionSpec(None)))
            )
            gathered_ref_grads.append(
                jax.lax.with_sharding_constraint(ref_grad, NamedSharding(mesh, PartitionSpec(None)))
            )
        jax.block_until_ready(gathered_grads)
        jax.block_until_ready(gathered_ref_grads)

    if args.enable_result_check and args.process_id == 0:
        assert_allclose(ref_output, output, dtype=jnp.bfloat16)
        for ref_grad, gathered_grad in zip(gathered_ref_grads, gathered_grads):
            assert_allclose(ref_grad, gathered_grad, dtype=jnp.bfloat16)


def dense_grad_parser(args):
    """Test settings."""
    parser = argparse.ArgumentParser(description="JAX Collective Dense Gradient Test")
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
        default=2048,
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
        default=None,
        help="Coordinator address for distributed initialization",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for distributed initialization",
    )
    parser.add_argument(
        "--num-devices-per-process",
        type=str,
        default=None,
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
        help="Local device IDs for distributed initialization",
    )
    return parser.parse_args(args)


class TestCollectiveDenseGradient(unittest.TestCase):
    """Collective Dense Gradient unittests"""

    # is_fp8_supported, fp8_reason = is_fp8_available(ScalingMode.DELAYED_TENSOR_SCALING)
    # is_mxfp8_supported, mxfp8_reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

    def setUp(self):
        """Set up test environment for pytest execution."""
        # Create args object with distributed parameters from pytest fixtures
        self.args = dense_grad_parser([])
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

    def tearDown(self):
        """Clean up after each test."""
        # Clear the mesh to prevent interference between tests
        jax.sharding.set_mesh(None)
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather(self):
        """Test Collective Dense Gradient with AllGather"""
        self.args.collective_type = "all_gather"
        run_dense_grad_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter(self):
        """Test Collective Dense Gradient with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_dense_grad_tests(self.args, self.mesh)


class TestCollectiveDenseGradientWithDP(unittest.TestCase):
    """Collective Dense Gradient with DP unittests"""

    def setUp(self):
        """Set up test environment for pytest execution."""
        # Create args object with distributed parameters from pytest fixtures
        self.args = dense_grad_parser([])
        self.args.coordinator_address = self.coordinator_address
        self.args.num_processes = self.num_processes
        self.args.process_id = self.process_id
        self.args.local_device_ids = self.local_device_ids
        self.args.num_devices_per_process = self.num_devices_per_process
        _initialize_distributed(self.args)
        # Create mesh once for all tests
        self.args.enable_data_parallel = True
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
        """Clean up after each test."""
        # Clear the mesh to prevent interference between tests
        jax.sharding.set_mesh(None)
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather_with_dp(self):
        """Test Collective Dense Gradient with AllGather"""
        self.args.collective_type = "all_gather"
        run_dense_grad_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter_with_dp(self):
        """Test Collective Dense Gradient with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_dense_grad_tests(self.args, self.mesh)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 7:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print("Usage: python test_dense_grad.py --coordinator-address <address> --num-processes <num> --process-id <id> [--local-device-ids <ids>] [other args]")
        print("Example: python test_dense_grad.py --coordinator-address localhost:1234 --num-processes 4 --process-id 0")
        print("Example: python test_dense_grad.py --coordinator-address localhost:1234 --num-processes 2 --process-id 0 --local-device-ids 0,1,2,3")
        sys.exit(1)

    args = dense_grad_parser(None)
    _initialize_distributed(args)
    run_dense_grad_tests(args, mesh=None)
