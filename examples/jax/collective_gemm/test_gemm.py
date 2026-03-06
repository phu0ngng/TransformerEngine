# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding

from common import (
    assert_allclose,
    dtype_tols,
    _initialize_distributed,
    _get_dp_and_tp_sizes,
    _create_mesh,
    DP_AXIS,
    TPSP_AXIS,
    PARAMS_KEY,
    cgemm_parser,
    get_quantization_recipe_from_name_string,
    get_scaling_mode_from_recipe_name,
)

import transformer_engine.jax.cpp_extensions as tex
from transformer_engine.jax.quantize import (
    autocast,
    is_scaling_mode_supported,
    QuantizerFactory,
    noop_quantizer_set,
)
from transformer_engine.jax.cpp_extensions.gemm import CollectiveOp
from transformer_engine.jax.sharding import MeshResource


def _get_operand_sharding(mesh, collective_op, is_with_dp):

    dp_axis = DP_AXIS if is_with_dp else None
    if collective_op == CollectiveOp.ALL_GATHER:
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, TPSP_AXIS, None))
        weight_sharding = NamedSharding(mesh, PartitionSpec(None, TPSP_AXIS))
        bias_sharding = NamedSharding(mesh, PartitionSpec(TPSP_AXIS))
        output_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, TPSP_AXIS))
    else:  # RS
        x_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, None, TPSP_AXIS))
        weight_sharding = NamedSharding(mesh, PartitionSpec(TPSP_AXIS, None))
        bias_sharding = NamedSharding(mesh, PartitionSpec(None))
        output_sharding = NamedSharding(mesh, PartitionSpec(dp_axis, TPSP_AXIS, None))

    return x_sharding, weight_sharding, bias_sharding, output_sharding


def _get_dp_and_tp_sizes(args):
    num_gpu = args.num_processes * args.num_devices_per_process
    if args.tensor_parallel_size is None:
        num_gpu_dp = 2 if args.enable_data_parallel else 1
        assert (
            num_gpu > 1 and num_gpu % num_gpu_dp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_tp = num_gpu // num_gpu_dp
    else:
        num_gpu_tp = args.tensor_parallel_size
        assert (
            num_gpu > 1 and num_gpu % num_gpu_tp == 0
        ), "Number of GPUs must be greater than 1 and divisible by number of data parallel GPUs"
        num_gpu_dp = num_gpu // num_gpu_tp
    return num_gpu_dp, num_gpu_tp


@partial(jax.jit, static_argnames=("contracting_dims", "collective_op", "output_sharding"))
def _jitted_cgemm(x, weight, bias, quantizer_set, contracting_dims, collective_op, output_sharding):
    output = tex.gemm(
        x,
        weight,
        bias=bias,
        contracting_dims=contracting_dims,
        collective_op=collective_op,
        quantizer_set=quantizer_set,
    )
    if output_sharding is not None:
        output = jax.lax.with_sharding_constraint(output, output_sharding)
    return output


def run_gemm_tests(args, mesh=None):
    """Execute GEMM tests."""
    print(args)

    # Initialize distributed with provided arguments
    _initialize_distributed(args)
    mesh = mesh or _create_mesh(args)

    # Structured test data for easy error detection:
    #   x[b, s, :]   = (s % 8 + 1)  — every element in row s has the same value in [1..8]
    #   weight[:, j] = (j % 8 + 1)  — every element in col j has the same value in [1..8]
    #   bias         = 0
    # Expected output: y[b, s, j] = hidden_in * (s%8+1) * (j%8+1)
    x_row_vals = (jnp.arange(args.seq_len, dtype=jnp.float32) % 8 + 1).astype(jnp.bfloat16)
    x = jnp.broadcast_to(
        x_row_vals[None, :, None], (args.batch_size, args.seq_len, args.hidden_in)
    )
    w_col_vals = (jnp.arange(args.hidden_out, dtype=jnp.float32) % 8 + 1).astype(jnp.bfloat16)
    weight = jnp.broadcast_to(
        w_col_vals[None, :], (args.hidden_in, args.hidden_out)
    )
    bias = jnp.zeros((args.hidden_out,), dtype=jnp.bfloat16)
    collective_op = (
        CollectiveOp.ALL_GATHER
        if args.collective_type == "all_gather"
        else CollectiveOp.REDUCE_SCATTER
    )

    use_fp8 = getattr(args, "use_fp8", False)
    recipe = get_quantization_recipe_from_name_string(args.quantize_recipe) if use_fp8 else None

    # autocast sets the global recipe (fwd/bwd dtypes) AND the global MeshResource
    # (via global_shard_guard) required for collective GEMM sharding axis resolution.
    with mesh, autocast(
        enabled=use_fp8,
        recipe=recipe,
        mesh_resource=MeshResource(dp_resource=DP_AXIS, tpsp_resource=TPSP_AXIS),
    ):
        # Build quantizer_set inside autocast so create_set() can read the global recipe
        # for correct fwd/bwd dtypes. autocast does not inject quantizers into raw
        # tex.gemm() calls, so we must pass quantizer_set explicitly.
        quantizer_set = QuantizerFactory.create_set() if use_fp8 else noop_quantizer_set
        print(f"Device mesh: {mesh}")

        x_sharding, weight_sharding, bias_sharding, output_sharding = _get_operand_sharding(
            mesh, collective_op, args.enable_data_parallel
        )
        x_sharded = jax.device_put(x, x_sharding)
        weight_sharded = jax.device_put(weight, weight_sharding)
        bias_sharded = jax.device_put(bias, bias_sharding)

        ref_output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            quantizer_set,
            contracting_dims=((2,), (0,)),
            collective_op=CollectiveOp.NONE,
            output_sharding=output_sharding,
        )
        output = _jitted_cgemm(
            x_sharded,
            weight_sharded,
            bias_sharded,
            quantizer_set,
            contracting_dims=((2,), (0,)),
            collective_op=collective_op,
            output_sharding=output_sharding,
        )
        assert (
            ref_output.sharding == output.sharding
        ), f"ref_output.sharding={ref_output.sharding}, output.sharding={output.sharding}"
        gathered_ref_output = jax.lax.with_sharding_constraint(
            ref_output, NamedSharding(mesh, PartitionSpec(None))
        )
        gathered_output = jax.lax.with_sharding_constraint(
            output, NamedSharding(mesh, PartitionSpec(None))
        )
        jax.block_until_ready(gathered_ref_output)
        jax.block_until_ready(gathered_output)

    if args.process_id == 0:
        tol_dtype = quantizer_set.x.q_dtype if use_fp8 else gathered_ref_output.dtype
        tols = dtype_tols(tol_dtype)

        # Expected output formula: y[b, s, j] = hidden_in * (s%8+1) * (j%8+1)
        expected = (
            args.hidden_in
            * (jnp.arange(args.seq_len) % 8 + 1).astype(jnp.float32)[:, None]
            * (jnp.arange(args.hidden_out) % 8 + 1).astype(jnp.float32)[None, :]
        )  # [seq_len, hidden_out]

        ref = jnp.array(gathered_ref_output[0], dtype=jnp.float32)  # [seq_len, hidden_out]
        out = jnp.array(gathered_output[0], dtype=jnp.float32)

        print("\n=== Output comparison (batch=0, rows 0..3, cols 0..7) ===")
        print(f"  expected:\n{expected[:4, :8]}")
        print(f"  ref (CollectiveOp.NONE):\n{ref[:4, :8]}")
        print(f"  output  (CollectiveOp.{args.collective_type.upper()}):\n{out[:4, :8]}")

        diff = jnp.abs(ref - out)
        mismatch_mask = diff > (tols["atol"] + tols["rtol"] * jnp.abs(ref))
        n_total = gathered_ref_output[0].size
        n_mismatch = int(jnp.sum(mismatch_mask))
        print(f"\n=== Mismatches: {n_mismatch} / {n_total} (tols: {tols}) ===")
        if 0 < n_mismatch <= 200:
            indices = jnp.argwhere(mismatch_mask)
            for s, j in indices[:50]:
                print(
                    f"  [0,{int(s):5d},{int(j):5d}]"
                    f"  expected={float(expected[s, j]):.2f}"
                    f"  ref={float(ref[s, j]):.4f}"
                    f"  got={float(out[s, j]):.4f}"
                    f"  diff={float(diff[s, j]):.4f}"
                )
        elif n_mismatch > 200:
            print(f"  (too many mismatches to list individually; showing first 50 rows, 8 cols)")
            print(f"  diff[0:50, 0:8]:\n{diff[:50, :8]}")

        if args.enable_result_check:
            assert_allclose(gathered_ref_output, gathered_output, dtype=tol_dtype)


class TestCollectiveGemmWithDP(unittest.TestCase):
    """Collective GEMM with DP unittests"""

    def setUp(self):
        self.args = cgemm_parser(
            "Collective GEMM test on multi-GPU with tensor parallelism"
        ).parse_args([])
        self.args.coordinator_address = self.coordinator_address
        self.args.num_processes = self.num_processes
        self.args.process_id = self.process_id
        self.args.local_device_ids = self.local_device_ids
        self.args.num_devices_per_process = self.num_devices_per_process
        # self.args.enable_data_parallel = True
        self.args.tensor_parallel_size = _get_dp_and_tp_sizes(self.args)[1]
        _initialize_distributed(self.args)
        self.mesh = _create_mesh(self.args)
        jax.sharding.set_mesh(self.mesh)
        self.args.enable_result_check = True
        os.environ["NVTE_JAX_ALL_REDUCE_IN_FP32"] = "1"

    def tearDown(self):
        os.environ.pop("NVTE_JAX_ALL_REDUCE_IN_FP32", None)

    def test_te_bf16_all_gather_with_dp(self):
        """Test Collective GEMM with AllGather"""
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_bf16_reduce_scatter_with_dp(self):
        """Test Collective GEMM with ReduceScatter"""
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)

    def test_te_delayed_scaling_fp8_all_gather_with_dp(self):
        """Test Collective GEMM with FP8 DelayedScaling + AllGather"""
        self.args.quantize_recipe = "DelayedScaling"
        is_supported, reason = is_scaling_mode_supported(
            get_scaling_mode_from_recipe_name(self.args.quantize_recipe)
        )
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_delayed_scaling_fp8_reduce_scatter_with_dp(self):
        """Test Collective GEMM with FP8 DelayedScaling + ReduceScatter"""
        self.args.quantize_recipe = "DelayedScaling"
        is_supported, reason = is_scaling_mode_supported(
            get_scaling_mode_from_recipe_name(self.args.quantize_recipe)
        )
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)

    def test_te_current_scaling_fp8_all_gather_with_dp(self):
        """Test Collective GEMM with FP8 Float8CurrentScaling + AllGather"""
        self.args.quantize_recipe = "Float8CurrentScaling"
        is_supported, reason = is_scaling_mode_supported(
            get_scaling_mode_from_recipe_name(self.args.quantize_recipe)
        )
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_current_scaling_fp8_reduce_scatter_with_dp(self):
        """Test Collective GEMM with FP8 Float8CurrentScaling + ReduceScatter"""
        self.args.quantize_recipe = "Float8CurrentScaling"
        is_supported, reason = is_scaling_mode_supported(
            get_scaling_mode_from_recipe_name(self.args.quantize_recipe)
        )
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)

    def test_te_mxfp8_all_gather_with_dp(self):
        """Test Collective GEMM with MXFP8BlockScaling + AllGather"""
        self.args.quantize_recipe = "MXFP8BlockScaling"
        is_supported, reason = is_scaling_mode_supported(get_scaling_mode_from_recipe_name(self.args.quantize_recipe))
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "all_gather"
        run_gemm_tests(self.args, self.mesh)

    def test_te_mxfp8_reduce_scatter_with_dp(self):
        """Test Collective GEMM with MXFP8BlockScaling + ReduceScatter"""
        self.args.quantize_recipe = "MXFP8BlockScaling"
        is_supported, reason = is_scaling_mode_supported(get_scaling_mode_from_recipe_name(self.args.quantize_recipe))
        if not is_supported:
            self.skipTest(reason)
        self.args.use_fp8 = True
        self.args.collective_type = "reduce_scatter"
        run_gemm_tests(self.args, self.mesh)

    # TODO: Enable when NVFP4BlockScaling + Collective GEMM is supported
    # def test_te_nvfp4_all_gather_with_dp(self):
    #     """Test Collective GEMM with NVFP4BlockScaling + AllGather"""
    #     self.args.quantize_recipe = "NVFP4BlockScaling"
    #     is_supported, reason = is_scaling_mode_supported(get_scaling_mode_from_recipe_name(self.args.quantize_recipe))
    #     if not is_supported:
    #         self.skipTest(reason)
    #     self.args.use_fp8 = True
    #     self.args.collective_type = "all_gather"
    #     run_gemm_tests(self.args, self.mesh)

    # TODO: Enable when NVFP4BlockScaling + Collective GEMM is supported
    # def test_te_nvfp4_reduce_scatter_with_dp(self):
    #     """Test Collective GEMM with NVFP4BlockScaling + ReduceScatter"""
    #     self.args.quantize_recipe = "NVFP4BlockScaling"
    #     is_supported, reason = is_scaling_mode_supported(get_scaling_mode_from_recipe_name(self.args.quantize_recipe))
    #     if not is_supported:
    #         self.skipTest(reason)
    #     self.args.use_fp8 = True
    #     self.args.collective_type = "reduce_scatter"
    #     run_gemm_tests(self.args, self.mesh)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:  # Need at least the 3 required distributed args
        print("Error: This script requires distributed initialization arguments.")
        print(
            "Usage: python test_gemm.py --coordinator-address <address> --num-processes <num>"
            " --process-id <id> [--local-device-ids <ids>] [other args]"
        )
        sys.exit(1)

    args = cgemm_parser("Collective GEMM test on multi-GPU with tensor parallelism").parse_args()
    _initialize_distributed(args)
    run_gemm_tests(args, mesh=None)
