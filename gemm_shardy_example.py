#!/usr/bin/env python3
"""
Specific example demonstrating sharding propagation issues with the exact configuration:

Input[batch, seq, hidden_in], PartitionSpec(data, tensor, None)
Weight[hidden_in, hidden_out], PartitionSpec(None, tensor)
output = gemm(Input, Weight, contracting_dim=((2,), (0,))
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import xla, mlir, batching
from jax.experimental.custom_partitioning import custom_partitioning, SdyShardingRule
from jax._src import dispatch
from jax.sharding import PartitionSpec
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
    Input[batch, seq, hidden_in] @ Weight[hidden_in, hidden_out]
    contracting_dim=((2,), (0,))  # hidden_in dimension
    """

    name = "te_gemm"
    multiple_results = True
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(lhs, rhs, contracting_dims):
        """Abstract evaluation for the specific GEMM configuration."""
        batch, seq, hidden_in = lhs.shape
        hidden_in_weight, hidden_out = rhs.shape
        out_shape = (batch, seq, hidden_out)
        out_dtype = jnp.result_type(lhs.dtype, rhs.dtype)
        return (jax.core.ShapedArray(shape=out_shape, dtype=out_dtype),)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        return GemmPrimitive.abstract(*args, **kwargs)

    @staticmethod
    def impl(lhs, rhs, contracting_dims):
        return GemmPrimitive.inner_primitive.bind(lhs, rhs, contracting_dims=contracting_dims)

    @staticmethod
    def lowering(ctx, lhs, rhs, contracting_dims):
        return (lhs,)       # no lowering, no calculation, just return lhs as the output

    @staticmethod
    def batcher(batched_args, batch_dims, contracting_dims):
        return GemmPrimitive.outer_primitive.bind(lhs, rhs, contracting_dims=contracting_dims)

    @staticmethod
    def _parse_operand_output_specs(arg_infos, contracting_dims):
        """
        Parse operand and output specs - copied from TE to demonstrate we can conditionally
        overwrite specs to trigger AG if we have access to inputs specs
        """
        lhs_specs, rhs_specs = map(get_padded_spec, arg_infos)

        lhs_ndim, rhs_ndim = map(len, (lhs_specs, rhs_specs))
        lhs_cdims, rhs_cdims = map(
            sanitize_dims, (lhs_ndim, rhs_ndim), contracting_dims
        )
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
            lhs_cspecs = tuple(s if s == reduce_spec else None for s in lhs_cspecs)
            rhs_cspecs = tuple(s if s == reduce_spec else None for s in rhs_cspecs)
            rhs_non_cspecs = tuple(
                None if spec in lhs_non_cspecs else spec for spec in rhs_non_cspecs
            )

        else:
            # Otherwise, require contracting dims of both operands to be unsharded
            lhs_cspecs = (None,) * len(lhs_cspecs)
            rhs_cspecs = (None,) * len(rhs_cspecs)
            # rhs_non_cspecs = tuple(None for _ in rhs_non_cspecs)

        lhs_non_cspecs = tuple(
            None if spec in rhs_non_cspecs else spec for spec in lhs_non_cspecs
        )

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
            (out_specs,),  # out_specs
            reduce_spec,  # reduce_spec
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
        print("--GemmPrimitive.infer_sharding_from_operands is called")

        (_, (out_specs,), *_) = GemmPrimitive._parse_operand_output_specs(
            arg_infos, contracting_dims
        )
        output_sharding = NamedSharding(mesh, PartitionSpec(*out_specs))

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

        print("--GemmPrimitive.partition is called")

        (lhs_specs, rhs_specs), (output_specs,), _ = GemmPrimitive._parse_operand_output_specs(
            arg_infos, contracting_dims
        )

        lhs_sharding = NamedSharding(mesh, PartitionSpec(*lhs_specs))
        rhs_sharding = NamedSharding(mesh, PartitionSpec(*rhs_specs))
        output_sharding = NamedSharding(mesh, PartitionSpec(*output_specs))

        def sharded_impl(lhs, rhs):
            return GemmPrimitive.impl(lhs, rhs, contracting_dims=contracting_dims)

        return mesh, sharded_impl, [output_sharding], (lhs_sharding, rhs_sharding)

    @staticmethod
    def shardy_sharding_rule(contracting_dims, mesh, value_types, result_types):
        """Sharding rule for Shardy."""
        del mesh, result_types

        print("--GemmPrimitive.shardy_sharding_rule is called")

        prefix = "TeGemm_"

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

        output_specs = [lhs_specs[i] for i in lhs_non_cdims] + [
            rhs_specs[i] for i in rhs_non_cdims
        ]
        # printed values
        # lhs_specs ['TeGemm_lhs_d0', 'TeGemm_lhs_d1', 'TeGemm_lhs_k0']
        # rhs_specs ['TeGemm_rhs_k0', 'TeGemm_rhs_d1']
        # output_specs ['TeGemm_lhs_d0', 'TeGemm_lhs_d1', 'TeGemm_rhs_d1']

        return SdyShardingRule(
            operand_mappings=(lhs_specs, rhs_specs),
            result_mappings=(output_specs,),
        )


def register_specific_gemm_primitive():
    """Register the specific GEMM primitive with JAX."""

    # Create inner primitive
    inner_p = core.Primitive(GemmPrimitive.name)
    dispatch.prim_requires_devices_during_lowering.add(inner_p)
    inner_p.multiple_results = GemmPrimitive.multiple_results
    inner_p.def_impl(partial(xla.apply_primitive, inner_p))
    inner_p.def_abstract_eval(GemmPrimitive.abstract)
    mlir.register_lowering(inner_p, GemmPrimitive.lowering, platform="cuda")
    GemmPrimitive.inner_primitive = inner_p

    # Create outer primitive with custom partitioning
    outer_p = core.Primitive(GemmPrimitive.name + "_wrapper")
    dispatch.prim_requires_devices_during_lowering.add(outer_p)
    outer_p.multiple_results = GemmPrimitive.multiple_results
    outer_p.def_impl(GemmPrimitive.impl)
    outer_p.def_abstract_eval(GemmPrimitive.outer_abstract)
    batching.primitive_batchers[outer_p] = GemmPrimitive.batcher

    # Register custom partitioning
    outer_p_lower = custom_partitioning(
        GemmPrimitive.impl, static_argnums=(2,)  # contracting_dims is static
    )
    outer_p_lower.def_partition(
        infer_sharding_from_operands=GemmPrimitive.infer_sharding_from_operands,
        partition=GemmPrimitive.partition,
        sharding_rule=GemmPrimitive.shardy_sharding_rule,
    )
    mlir.register_lowering(
        outer_p,
        mlir.lower_fun(outer_p_lower, multiple_results=GemmPrimitive.multiple_results),
    )
    GemmPrimitive.outer_primitive = outer_p


# Register the primitive
register_specific_gemm_primitive()

# Create a 2x2 mesh using only 4sGPUs
available_devices = jax.devices()
assert len(available_devices) >= 4, f"Need at least 4 GPUs"
devices = available_devices[:4]
devices = np.array(devices).reshape((2, 2))
mesh = Mesh(devices, ("tensor", "data"))
jax.sharding.set_mesh(mesh)

# Create test data with the exact configuration
batch_size = 32
seq_len = 32
hidden_in = 32
hidden_out = 64 # need to be x2 hidden_in so that return lhs_shape = output_shape in lowering

# Input: [batch, seq, hidden_in]
input_tensor = jnp.ones((batch_size, seq_len, hidden_in), dtype=jnp.bfloat16)

# Weight: [hidden_in, hidden_out]
weight_tensor = jnp.ones((hidden_in, hidden_out), dtype=jnp.bfloat16)

# print("=== Specific GEMM Configuration Test ===")
# print(f"Mesh: {mesh}")
# print(f"Input shape: {input_tensor.shape}")
# print(f"Weight shape: {weight_tensor.shape}")
# print(f"Contracting dims: ((2,), (0,))")
# print(f"Expected output shape: ({batch_size}, {seq_len}, {hidden_out})")

# Define the sharding configurations
input_sharding = PartitionSpec("data", "tensor", None)
weight_sharding = PartitionSpec(None, "tensor")
expected_output_sharding = PartitionSpec("data", None, 'tensor')

print(f"Input sharding: {input_sharding}")
print(f"Weight sharding: {weight_sharding}")
print(f"Expected output sharding: {expected_output_sharding}")

@jax.custom_vjp
def _gemm(x, w):
    output, _ = _gemm_fwd(x, w)
    return output

def _gemm_fwd(x, w):
    output = GemmPrimitive.outer_primitive.bind(x, w, contracting_dims=((2,), (0,)))
    # output = jax.lax.with_sharding_constraint(output, expected_output_sharding)  # <------ WAR if known output sharding constraint is available
    jax.debug.print("\nOutput sharding")
    jax.debug.inspect_array_sharding(output, callback=print)
    return output, None

def _gemm_bwd(ctx, grad):
    return None, None

_gemm.defvjp(_gemm_fwd, _gemm_bwd)

def execute_test(shardy_enabled=False):
    """Execute the GEMM test with or without Shardy."""
    print(f"\n{'='*60}")
    print(f"Running test with Shardy {'ON' if shardy_enabled else 'OFF'}")
    print(f"{'='*60}")

    # Set Shardy configuration
    if shardy_enabled:
        jax.config.update('jax_use_shardy_partitioner', True)
        print("Shardy partitioner enabled")
    else:
        jax.config.update('jax_use_shardy_partitioner', False)
        print("Shardy partitioner disabled")

    with mesh:
        # Shard the inputs
        input_sharded = jax.device_put(input_tensor, input_sharding)
        weight_sharded = jax.device_put(weight_tensor, weight_sharding)

        try:
            jitted_gemm = jax.jit(jax.value_and_grad(lambda x, w: jax.numpy.mean(_gemm(x, w)[0]), argnums=(0, 1)),
                                  in_shardings=[input_sharding, weight_sharding],
                                  #out_shardings=expected_output_sharding,
                                  )
            result, *_ = jitted_gemm(input_sharded, weight_sharded)

        except Exception as e:
            print(f"GEMM failed: {e}")
            import traceback
            traceback.print_exc()

# Execute test twice - first without Shardy, then with Shardy
execute_test(shardy_enabled=False)
execute_test(shardy_enabled=True)
