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
from jax.experimental.custom_partitioning import custom_partitioning, SdyShardingRule
from jax._src import dispatch
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
import numpy as np
from typing import Union, Sequence, Iterable


def sanitize_dims(ndim: int, dims: Union[int, Sequence[int]]) -> Sequence[int]:
    """Convert relative (negative) indexes to absolute dimension numbers."""
    dims_ = dims if isinstance(dims, Iterable) else (dims,)
    if len(dims_) == 0:
        return dims_
    return tuple(ndim + dim if dim < 0 else dim for dim in dims_ if dim is not None)


def get_padded_spec(arg_info):
    """Get padded spec for partitioning from arguments' information."""
    if arg_info.sharding is None:
        return (None,) * arg_info.ndim
    return arg_info.sharding.spec


class GemmPrimitive:
    """
    GEMM primitive for the specific configuration:
    Input[batch, seq, hidden_in] @ Weight[hidden_in, act_num, hidden_out]
    contracting_dim=((2,), (0,))  # hidden_in dimension
    """
    
    name = "specific_gemm"
    multiple_results = False
    
    @staticmethod
    def abstract(lhs, rhs, contracting_dims):
        """Abstract evaluation for the specific GEMM configuration."""
        batch, seq, hidden_in = lhs.shape
        hidden_in_weight, act_num, hidden_out = rhs.shape
        
        # Validate contracting dimensions
        assert contracting_dims == ((2,), (0,)), f"Expected contracting_dims=((2,), (0,)), got {contracting_dims}"
        assert hidden_in == hidden_in_weight, f"Contracting dimension mismatch: {hidden_in} != {hidden_in_weight}"
        
        # Output shape: [batch, seq, act_num, hidden_out]
        out_shape = (batch, seq, act_num, hidden_out)
        out_dtype = jnp.result_type(lhs.dtype, rhs.dtype)
        return jax.core.ShapedArray(shape=out_shape, dtype=out_dtype)
    
    @staticmethod
    def impl(lhs, rhs, contracting_dims):
        """Implementation - compute the actual GEMM operation."""
        # For demonstration, we'll do a simple reshape and broadcast
        # In a real implementation, this would call the actual GEMM kernel
        batch, seq, hidden_in = lhs.shape
        hidden_in_weight, act_num, hidden_out = rhs.shape
        
        # Validate contracting dimensions
        assert contracting_dims == ((2,), (0,)), f"Expected contracting_dims=((2,), (0,)), got {contracting_dims}"
        assert hidden_in == hidden_in_weight, f"Contracting dimension mismatch: {hidden_in} != {hidden_in_weight}"
        
        # Simulate GEMM: input[batch, seq, hidden_in] @ weight[hidden_in, act_num, hidden_out]
        # -> output[batch, seq, act_num, hidden_out]
        # For this demo, we'll just reshape and broadcast
        output = lhs[:, :, None, :] * rhs[None, None, :, :]
        return output
    
    @staticmethod
    def lowering(ctx, lhs, rhs, contracting_dims):
        """Lowering - minimal implementation."""
        return mlir.return_op([lhs])
    
    @staticmethod
    def batcher(batched_args, batch_dims, contracting_dims):
        """Batching rule."""
        lhs, rhs = batched_args
        lhs_bdim, rhs_bdim = batch_dims
        
        if lhs_bdim is not None or rhs_bdim is not None:
            raise ValueError("Batching not supported in this example")
        
        return GemmPrimitive.bind(lhs, rhs, ((2,), (0,))), None
    
    @staticmethod
    def _parse_operand_output_specs(arg_infos, contracting_dims):
        """
        Parse operand and output specs - exact implementation from TransformerEngine.
        """
        lhs_specs, _, rhs_specs, *_ = map(get_padded_spec, arg_infos)

        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims = map(sanitize_dims, (lhs_ndim, rhs_ndim), contracting_dims)
        lhs_non_cdims, rhs_non_cdims = map(
            lambda ndim, cdims: tuple(i for i in range(ndim) if i not in cdims),
            (lhs_ndim, rhs_ndim),
            (lhs_cdims, rhs_cdims),
        )
        lhs_non_cspecs, lhs_cspecs, rhs_non_cspecs, rhs_cspecs = map(
            lambda specs, dims: tuple(specs[i] for i in dims),
            (lhs_specs, lhs_specs, rhs_specs, rhs_specs),
            (lhs_non_cdims, lhs_cdims, rhs_non_cdims, rhs_cdims),
        )

        reduce_spec = None
        for l in lhs_cspecs:
            for r in rhs_cspecs:
                if l is not None and l == r:
                    assert reduce_spec is None, "Multiple reduce dimension is detected!"
                    reduce_spec = l

        if reduce_spec is not None:
            # Other non-reduce cdims (if exists) need to be unsharded
            lhs_cspecs = tuple(s if s == reduce_spec else None for s in lhs_cspecs)
            rhs_cspecs = tuple(s if s == reduce_spec else None for s in rhs_cspecs)

            # Non-contracting dims of RHS always needs to be gathered, i.e. for TP + activation_hidden
            # No batch-dim check needed as `rhs_non_cspecs` never contains batch-dim.
            # In `rhs_specs`, the batch dim appears only in Wgrad GEMM under `rhs_cspecs`.
            rhs_non_cspecs = tuple(
                None if spec in lhs_non_cspecs else spec for spec in rhs_non_cspecs
            )

        else:
            # Otherwise, require contracting dims of both operands to be unsharded
            lhs_cspecs = (None,) * len(lhs_cspecs)
            rhs_cspecs = (None,) * len(rhs_cspecs)

            # Non-contracting dims of RHS always needs to be gathered along the FSDP axis
            # For our simplified version, we'll just gather all non-contracting dims
            # rhs_non_cspecs = tuple(None for _ in rhs_non_cspecs)

        # Non-contracting dims of LHS to be gathered along the SP axis.
        # Minor note: This causes MaxText TP (= Megatron TP + activation_hidden sharding) gathering x for
        # dW1 = x^T * dY1 which is unexpected. This is a known issue and no solution has found yet.
        lhs_non_cspecs = tuple(None if spec in rhs_non_cspecs else spec for spec in lhs_non_cspecs)

        out_specs = lhs_non_cspecs + rhs_non_cspecs

        # specs = merge(cspecs, non_cspecs)
        lhs_specs, rhs_specs = map(
            lambda cdims, cspecs, non_cspecs: (
                cspecs + non_cspecs if cdims[0] == 0 else non_cspecs + cspecs
            ),
            (lhs_cdims, rhs_cdims),
            (lhs_cspecs, rhs_cspecs),
            (lhs_non_cspecs, rhs_non_cspecs),
        )

          return (
            (lhs_specs, rhs_specs),  # arg_specs
            (out_specs,),              # out_specs
            reduce_spec,              # reduce_spec
        )
    
    @staticmethod
    def infer_sharding_from_operands(
        contracting_dims,
        mesh,
        arg_infos,
        result_infos,
    ):
        """
        Infer output sharding for the specific configuration.
        
        This follows the TransformerEngine pattern where:
        - arg_infos contains information about input arguments
        - We parse the operand specs and determine output sharding
        """
        del result_infos
        
        # Parse operand specs
        (_, (out_specs,), _) = GemmPrimitive._parse_operand_output_specs(arg_infos, contracting_dims)
        
        # Create NamedSharding for output
        output_sharding = NamedSharding(mesh, P(*out_specs))
        
        print(f"  Inferred output sharding: {output_sharding}")
        return [output_sharding]
    
    @staticmethod
    def partition(
        contracting_dims,
        mesh,
        arg_infos,
        result_infos,
    ):
        """Partition function following TransformerEngine's approach."""
        del result_infos
        
        print(f"  Partition called")
        
        # Parse operand and output specs
        (arg_specs, out_specs, _) = GemmPrimitive._parse_operand_output_specs(arg_infos, contracting_dims)
        
        # Create shardings
        lhs_specs, rhs_specs = arg_specs
        output_specs = out_specs[0]
        
        lhs_sharding = NamedSharding(mesh, P(*lhs_specs))
        rhs_sharding = NamedSharding(mesh, P(*rhs_specs))
        output_sharding = NamedSharding(mesh, P(*output_specs))
        
        # Define sharded implementation
        def sharded_impl(lhs, rhs):
            return GemmPrimitive.impl(lhs, rhs, contracting_dims)
        
        return mesh, sharded_impl, [output_sharding], [input_sharding, weight_sharding]
    
    @staticmethod
    def shardy_sharding_rule(contracting_dims, mesh, value_types, result_types):
        """Sharding rule for Shardy."""
        del mesh, result_types
        
        prefix = "SpecificGemm_"
        
        # Extract input and weight types
        lhs_type, rhs_type = value_types[0], value_types[1]
        lhs_ndim, rhs_ndim = len(lhs_type.shape), len(rhs_type.shape)
        
        # Parse contracting dimensions
        lhs_cdims, rhs_cdims = contracting_dims
        
        def generate_operand_specs(name, ndim, cdims):
            """Generate specs for an operand based on its shape and contracting dims."""
            specs = []
            for i in range(ndim):
                if i in cdims:
                    # Contracting dimension - use k0, k1, etc.
                    dim_idx = cdims.index(i)
                    specs.append(f"{prefix}{name}_k{dim_idx}")
                else:
                    # Non-contracting dimension - use operand-specific names
                    specs.append(f"{prefix}{name}_d{i}")
            return specs
        
        # Generate specs for both operands
        lhs_specs = generate_operand_specs("lhs", lhs_ndim, lhs_cdims)
        rhs_specs = generate_operand_specs("rhs", rhs_ndim, rhs_cdims)
        
        # Generate output specs: non-contracting dims from input + non-contracting dims from weight
        lhs_non_cdims = [i for i in range(lhs_ndim) if i not in lhs_cdims]
        rhs_non_cdims = [i for i in range(rhs_ndim) if i not in rhs_cdims]
        
        output_specs = (
            [lhs_specs[i] for i in lhs_non_cdims] + 
            [rhs_specs[i] for i in rhs_non_cdims]
        )
        # lhs_spec = (SpecificGemm_lhs_d0, SpecificGemm_lhs_d1, SpecificGemm_lhs_k0)
        # rhs_spec = (SpecificGemm_rhs_k0, SpecificGemm_rhs_d1, SpecificGemm_rhs_d2)
        # output_spec = (SpecificGemm_lhs_d0, SpecificGemm_lhs_d1, SpecificGemm_rhs_d1, SpecificGemm_rhs_d2)
        
        return SdyShardingRule(
            operand_mappings=(lhs_specs, rhs_specs),
            result_mappings=(output_specs,),
        )
    
    @staticmethod
    def bind(lhs, rhs, contracting_dims=((2,), (0,))):
        """Bind the primitive."""
        return GemmPrimitive.outer_primitive.bind(lhs, rhs, contracting_dims)


def register_specific_gemm_primitive():
    """Register the specific GEMM primitive with JAX."""
    
    # Create inner primitive
    inner_p = core.Primitive(GemmPrimitive.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = GemmPrimitive.multiple_results
    inner_p.def_impl(GemmPrimitive.impl)
    inner_p.def_abstract_eval(GemmPrimitive.abstract)
    mlir.register_lowering(inner_p, GemmPrimitive.lowering, platform="cuda")
    GemmPrimitive.inner_primitive = inner_p
    
    # Create outer primitive with custom partitioning
    outer_p = core.Primitive(GemmPrimitive.name + "_wrapper")
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = GemmPrimitive.multiple_results
    outer_p.def_impl(GemmPrimitive.impl)
    outer_p.def_abstract_eval(GemmPrimitive.abstract)
    batching.primitive_batchers[outer_p] = GemmPrimitive.batcher
    
    # Register custom partitioning
    outer_p_lower = custom_partitioning(
        GemmPrimitive.impl,
        static_argnums=(2,)  # contracting_dims is static
    )
    outer_p_lower.def_partition(
        infer_sharding_from_operands=GemmPrimitive.infer_sharding_from_operands,
        partition=GemmPrimitive.partition,
        sharding_rule=GemmPrimitive.shardy_sharding_rule,
    )
    mlir.register_lowering(
        outer_p,
        mlir.lower_fun(outer_p_lower, multiple_results=GemmPrimitive.multiple_results)
    )
    GemmPrimitive.outer_primitive = outer_p


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
