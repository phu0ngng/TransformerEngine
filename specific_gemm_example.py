#!/usr/bin/env python3
"""
Specific example demonstrating sharding propagation issues with the exact configuration:

Input[batch, seq, hidden_in], PartitionSpec(None, tensor, None)
Weight[hidden_in, act_num, hidden_out], PartitionSpec(None, None, tensor)
output = gemm(Input, Weight, contracting_dim=((2,), (0,))
"""

import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import xla, mlir, batching
from jax.experimental.custom_partitioning import custom_partitioning
from jax._src import dispatch
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import numpy as np


class SpecificGemmPrimitive:
    """
    GEMM primitive for the specific configuration:
    Input[batch, seq, hidden_in] @ Weight[hidden_in, act_num, hidden_out]
    contracting_dim=((2,), (0,))  # hidden_in dimension
    """

    name = "specific_gemm"
    multiple_results = False

    @staticmethod
    def abstract(input_tensor, weight_tensor, contracting_dims):
        """Abstract evaluation for the specific GEMM configuration."""
        batch, seq, hidden_in = input_tensor.shape
        hidden_in_weight, act_num, hidden_out = weight_tensor.shape

        # Validate contracting dimensions
        assert contracting_dims == ((2,), (0,)), f"Expected contracting_dims=((2,), (0,)), got {contracting_dims}"
        assert hidden_in == hidden_in_weight, f"Contracting dimension mismatch: {hidden_in} != {hidden_in_weight}"

        # Output shape: [batch, seq, act_num, hidden_out]
        out_shape = (batch, seq, act_num, hidden_out)
        out_dtype = jnp.result_type(input_tensor.dtype, weight_tensor.dtype)
        return jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)

    @staticmethod
    def impl(input_tensor, weight_tensor, contracting_dims):
        """Implementation - just return input for demonstration."""
        print(f"SpecificGemmPrimitive.impl called:")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Weight shape: {weight_tensor.shape}")
        print(f"  Contracting dims: {contracting_dims}")
        return input_tensor[:, :, None, :]  # Expand to match output shape

    @staticmethod
    def lowering(ctx, input_tensor, weight_tensor, contracting_dims):
        """Lowering - minimal implementation."""
        return mlir.return_op([input_tensor])

    @staticmethod
    def batcher(batched_args, batch_dims, contracting_dims):
        """Batching rule."""
        del contracting_dims
        input_tensor, weight_tensor = batched_args
        input_bdim, weight_bdim = batch_dims

        if input_bdim is not None or weight_bdim is not None:
            raise ValueError("Batching not supported in this example")

        return SpecificGemmPrimitive.bind(input_tensor, weight_tensor, ((2,), (0,))), None

    @staticmethod
    def infer_sharding_from_operands(contracting_dims, args_info, result_info):
        """
        Infer output sharding for the specific configuration.
        """
        input_sharding = get_padded_spec(args_info[0])
        weight_sharding = get_padded_spec(args_info[1])

        lhs_cdims, rhs_cdims = (contracting_dims)
        # TODO

        print(f"  Inferred output sharding: {output_sharding}")
        return output_sharding

    @staticmethod
    def partition(contracting_dims, mesh, args_info, result_info):
        """Partition function."""
        print(f"  Partition called with spec: {spec}")

        output_sharding =
        arg_shardings = tuple(arg_i.sharding for arg_i in arg_infos)
        def sharded_impl(input_tensor, weight_tensor):
            output = SpecificGemmPrimitive.impl(input_tensor, weight_tensor, contracting_dims=contracting_dims)

        return mesh, sharded_impl, output_sharding, arg_shardings

    @staticmethod
    def shardy_sharding_rule(contracting_dims, mesh, value_types, result_types):
        """Sharding rule for Shardy."""
        return "... -> ..."

    @staticmethod
    def bind(input_tensor, weight_tensor, contracting_dims=((2,), (0,))):
        """Bind the primitive."""
        return SpecificGemmPrimitive.outer_primitive.bind(input_tensor, weight_tensor, contracting_dims)


def register_specific_gemm_primitive():
    """Register the specific GEMM primitive with JAX."""

    # Create inner primitive
    inner_p = core.Primitive(SpecificGemmPrimitive.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = SpecificGemmPrimitive.multiple_results
    inner_p.def_impl(SpecificGemmPrimitive.impl)
    inner_p.def_abstract_eval(SpecificGemmPrimitive.abstract)
    mlir.register_lowering(inner_p, SpecificGemmPrimitive.lowering, platform="cuda")
    SpecificGemmPrimitive.inner_primitive = inner_p

    # Create outer primitive with custom partitioning
    outer_p = core.Primitive(SpecificGemmPrimitive.name + "_wrapper")
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = SpecificGemmPrimitive.multiple_results
    outer_p.def_impl(SpecificGemmPrimitive.impl)
    outer_p.def_abstract_eval(SpecificGemmPrimitive.abstract)
    batching.primitive_batchers[outer_p] = SpecificGemmPrimitive.batcher

    # Register custom partitioning
    outer_p_lower = custom_partitioning(
        SpecificGemmPrimitive.impl,
        static_argnums=(2,)  # contracting_dims is static
    )
    outer_p_lower.def_partition(
        infer_sharding_from_operands=SpecificGemmPrimitive.infer_sharding_from_operands,
        partition=SpecificGemmPrimitive.partition,
        sharding_rule=SpecificGemmPrimitive.shardy_sharding_rule,
    )
    mlir.register_lowering(
        outer_p,
        mlir.lower_fun(outer_p_lower, multiple_results=SpecificGemmPrimitive.multiple_results)
    )
    SpecificGemmPrimitive.outer_primitive = outer_p



# Register the primitive
register_specific_gemm_primitive()

# Create a 2x2 mesh for tensor parallelism
devices = mesh_utils.create_device_mesh((2, 2))
mesh = Mesh(devices, ('tensor', 'data'))

# Create test data with the exact configuration
batch_size = 4
seq_len = 8
hidden_in = 6
act_num = 3
hidden_out = 10

# Input: [batch, seq, hidden_in]
input_tensor = jnp.ones((batch_size, seq_len, hidden_in), dtype=jnp.float32)

# Weight: [hidden_in, act_num, hidden_out]
weight_tensor = jnp.ones((hidden_in, act_num, hidden_out), dtype=jnp.float32)

print("=== Specific GEMM Configuration Test ===")
print(f"Mesh: {mesh}")
print(f"Input shape: {input_tensor.shape}")
print(f"Weight shape: {weight_tensor.shape}")
print(f"Contracting dims: ((2,), (0,))  # hidden_in dimension")
print(f"Expected output shape: ({batch_size}, {seq_len}, {act_num}, {hidden_out})")

# Define the sharding configurations
input_sharding = P(None, 'tensor', None)  # PartitionSpec(None, tensor, None)
weight_sharding = P(None, None, 'tensor')  # PartitionSpec(None, None, tensor)

print(f"\nInput sharding: {input_sharding}")
print(f"Weight sharding: {weight_sharding}")

with mesh:
    # Shard the inputs
    input_sharded = jax.device_put(input_tensor, input_sharding)
    weight_sharded = jax.device_put(weight_tensor, weight_sharding)

    print(f"\n--- Testing GEMM with specific configuration ---")

    # Call our primitive
    try:
        result = SpecificGemmPrimitive.bind(input_sharded, weight_sharded)
        print(f"GEMM succeeded!")
        print(f"Result shape: {result.shape}")

        # Check the actual sharding of the result
        result_sharding = jax.devices()[0].addressable_shards(result)[0].sharding
        print(f"Result sharding: {result_sharding}")

    except Exception as e:
        print(f"GEMM failed: {e}")
        import traceback
        traceback.print_exc()
