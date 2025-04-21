# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX te modules"""

from typing import Tuple, Sequence, Union, Dict, List
from functools import partial, reduce
import operator
import math
import jax
import jax.numpy as jnp
from transformer_engine_jax import get_device_compute_capability

from .base import BasePrimitive, register_primitive

from ..quantize import (
    ScaledTensor,
    GroupedScaledTensor1x,
    ScalingMode,
    Quantizer,
    GroupedQuantizer,
    QuantizeConfig,
    QuantizerSet,
    noop_quantizer_set,
)


__all__ = ["gemm", "grouped_gemm"]


num_cublas_streams = 4


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if get_device_compute_capability(0) >= 90:
        return 33_554_432
    return 4_194_304


def is_gemm_with_all_layouts_supported() -> False:
    """Return True if using blackwell, False otherwise."""
    return get_device_compute_capability(0) >= 100


class GroupedGemmPrimitive(BasePrimitive):
    """
    Primitive for grouped GEMM
    """

    name = "te_grouped_gemm_ffi"
    multiple_results = True
    impl_static_args = ()
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(lhs_data,
                 lhs_scale_inv,
                 rhs_data,
                 rhs_scale_inv,
                 bias,
                 group_sizes,
                 group_offset,
                 M,
                 N,
                 K,
                 lhs_is_trans,
                 rhs_is_trans,
                 scaling_mode,
                 out_dtype,
                 has_bias,
                 ):
        """
            scaling_mode: Scaling mode for the GEMM operations.
            out_dtype: Data type of the output tensors.
            has_bias: Boolean indicating if bias tensors are provided.

        Returns:
           1D flattened array of Grouped GEMM outputs
        """
        del scaling_mode
        # TODO(Phuong): move some shape checks from Cpp to here
        workspace_size = get_cublas_workspace_size_bytes() * num_cublas_streams
        workspace_aval = jax.core.ShapedArray(shape=(workspace_size,), dtype=jnp.uint8)
        out_aval = jax.core.ShapedArray(shape=(M, N), dtype=out_dtype)
        return (out_aval, workspace_aval)

    @staticmethod
    def outer_abstract(*args, **kwargs):
        (out_aval, _) = GroupedGemmPrimitive.abstract(*args, **kwargs)
        return out_aval

    @staticmethod
    def lowering(ctx, *args, M, N, K, lhs_is_trans, rhs_is_trans,
                 scaling_mode, out_dtype, has_bias):
        del out_dtype
        return jax.ffi.ffi_lowering(GroupedGemmPrimitive.name)(
            ctx,
            *args,
            M=M,
            N=N,
            K=K,
            lhs_is_trans=lhs_is_trans,
            rhs_is_trans=rhs_is_trans,
            scaling_mode=scaling_mode.value,
            has_bias=has_bias,
        )

    @staticmethod
    def impl(*args, **kwargs):
        assert GroupedGemmPrimitive.inner_primitive is not None
        out, _ = GroupedGemmPrimitive.inner_primitive.bind(*args, **kwargs)  # exclude wkspace
        return out


register_primitive(GroupedGemmPrimitive)


def _shape_normalization(x, dimension_numbers, already_transposed: bool = False):
    orig_order = list(range(x.ndim))
    contracting_dims, batch_dims = dimension_numbers
    contracting_order = [d for d in orig_order if d in contracting_dims]
    batch_order = [d for d in orig_order if d in batch_dims]
    non_contracting_order = [
        d for d in orig_order if d not in contracting_dims and d not in batch_dims
    ]
    batch_shape = [x.shape[d] for d in batch_order]
    rows_shape = [x.shape[d] for d in non_contracting_order]
    cols_shape = [x.shape[d] for d in contracting_order]
    new_order = batch_order + non_contracting_order + contracting_order
    rows, cols, batches = (
        reduce(operator.mul, rows_shape, 1),
        reduce(operator.mul, cols_shape, 1),
        reduce(operator.mul, batch_shape, 1),
    )
    # Remove this transpose when non-TN dot is supported
    if not already_transposed:
        t = jnp.transpose(x, new_order)
    else:
        t = x
    return jnp.reshape(t, (batches, rows, cols))


def _calculate_remaining_shape(shape, contracting_dims):
    return tuple(shape[dim] for dim in range(len(shape)) if dim not in contracting_dims)


def _dequantize(x, scale_inv, dq_dtype):
    return x.astype(dq_dtype) * scale_inv.astype(dq_dtype)


# Apply jit to guarantee correctness of FP8 GEMM.
@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
    ),
)
def __jitted_jax_gemm_delayed_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision):
    # Need to hard-code the dequantize here instead of calling lhs.dequantize() for pattern matching
    lhs_dq = _dequantize(lhs.data, lhs.scale_inv, lhs.dq_dtype)
    rhs_dq = _dequantize(rhs.data, rhs.scale_inv, rhs.dq_dtype)

    # Reshape + Transpose
    # [..., M, K] -> [B, M, K]
    # [..., K, M] -> [B, M, K]
    lhs_3d = _shape_normalization(lhs_dq, lhs_dn, lhs.data_layout == "N")
    rhs_3d = _shape_normalization(rhs_dq, rhs_dn, rhs.data_layout == "T")

    dim_nums = (((2,), (2,)), ((0,), (0,)))
    out_3d = jax.lax.dot_general(
        lhs_3d, rhs_3d, dim_nums, precision=precision, preferred_element_type=lhs.dq_dtype
    )
    return out_3d


def _jax_gemm_delayed_scaling_fp8(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """FP8 GEMM for XLA pattern match"""
    assert (
        rhs.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING
    ), "rhs does not have delayed tensor scaling mode"

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums
    if lhs.data_layout == "T":
        lhs_contract = tuple((lhs.data.ndim - 1 - i) % lhs.data.ndim for i in lhs_contract)
    if rhs.data_layout == "T":
        rhs_contract = tuple((rhs.data.ndim - 1 - i) % rhs.data.ndim for i in rhs_contract)

    lhs_dn = (lhs_contract, lhs_batch)
    rhs_dn = (rhs_contract, rhs_batch)

    lhs_remain_shape = _calculate_remaining_shape(lhs.data.shape, lhs_contract)
    rhs_remain_shape = _calculate_remaining_shape(rhs.data.shape, rhs_contract)

    precision = (
        jax.lax.Precision.HIGHEST if QuantizeConfig.FP8_2X_ACC_FPROP else jax.lax.Precision.DEFAULT
    )
    out_3d = __jitted_jax_gemm_delayed_scaling_fp8(lhs, rhs, lhs_dn, rhs_dn, precision)

    # Reshape [B, M, N] -> [..., M, N]
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm_mxfp8_1d(
    lhs: ScaledTensor, rhs: ScaledTensor, dim_nums: Tuple[Tuple[Sequence[int], Sequence[int]]]
):
    """
    JAX GEMM for MXFP8 via scaled_matmul
    """
    assert (
        rhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING
    ), "rhs does not have MXFP8 1D scaling mode"
    from jax._src.cudnn.scaled_matmul_stablehlo import scaled_matmul_wrapper

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dim_nums

    expected_lhs_is_colwise = lhs_contract[-1] != lhs.data.ndim - 1
    expected_rhs_is_colwise = rhs_contract[-1] != rhs.data.ndim - 1
    assert lhs.is_colwise is expected_lhs_is_colwise, (
        f"LHS with unexpected quantize dimension.\nExpect is_colwise={expected_lhs_is_colwise}, got"
        f" {lhs.is_colwise}"
    )
    assert rhs.is_colwise is expected_rhs_is_colwise, (
        f"RHS with unexpected quantize dimension.\nExpect is_colwise={expected_rhs_is_colwise}, got"
        f" {rhs.is_colwise}"
    )

    # Reshape + Transpose (if needed)
    # [..., M, K] -> [1, reduce(..., M), K]
    # [..., K, M] -> [1, reduce(..., M), K]
    lhs_3d = _shape_normalization(lhs.data, (lhs_contract, lhs_batch))
    rhs_3d = _shape_normalization(rhs.data, (rhs_contract, rhs_batch))
    lhs_scale_3d = _shape_normalization(lhs.scale_inv, (lhs_contract, lhs_batch))
    rhs_scale_3d = _shape_normalization(rhs.scale_inv, (rhs_contract, rhs_batch))

    # Slice out the padding as scaled_matmul does not support padded scales yet
    lhs_scale_3d = jnp.asarray(lhs_scale_3d[:, : lhs_3d.shape[1], : int(lhs_3d.shape[2] / 32)])
    rhs_scale_3d = jnp.asarray(rhs_scale_3d[:, : rhs_3d.shape[1], : int(rhs_3d.shape[2] / 32)])

    # JAX scaled_matmul only supports NT now (TN-gemm)
    # * Expected shape:
    # * lhs_data  (B, M, K)           * rhs_data  (B, N, K)
    # * lhs_scale (B, M, K_block)     * rhs_scale (B, N, K_block)
    out_3d = scaled_matmul_wrapper(
        lhs_3d, rhs_3d, lhs_scale_3d, rhs_scale_3d, preferred_element_type=lhs.dq_dtype
    )
    # Reshape [1, reduce(..., M), N] -> [..., M, N]
    lhs_remain_shape = tuple(
        lhs.data.shape[dim] for dim in range(len(lhs.data.shape)) if dim not in lhs_contract
    )
    rhs_remain_shape = tuple(
        rhs.data.shape[dim] for dim in range(len(rhs.data.shape)) if dim not in rhs_contract
    )
    out = out_3d.reshape(*lhs_remain_shape, *rhs_remain_shape)
    return out


def _jax_gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: Dict["str", Quantizer] = noop_quantizer_set,
) -> jnp.ndarray:
    """
    FP8 GEMM via JAX
    """

    dim_nums = (contracting_dims, ((), ()))

    def _jax_gemm_fp8_impl(lhs, rhs):

        if lhs.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING:
            return _jax_gemm_delayed_scaling_fp8(lhs, rhs, dim_nums)

        if lhs.scaling_mode == ScalingMode.MXFP8_1D_SCALING:
            return _jax_gemm_mxfp8_1d(lhs, rhs, dim_nums)

        raise NotImplementedError("Unsupported ScalingMode: {lhs.scaling_mode}")

    if isinstance(lhs, ScaledTensor) and isinstance(rhs, ScaledTensor):
        return _jax_gemm_fp8_impl(lhs, rhs)

    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor):
        if quantizer_set != noop_quantizer_set:
            assert type(quantizer_set.x) is type(quantizer_set.kernel)
            (((lhs_contract_dim,), (rhs_contract_dim,)), _) = dim_nums
            lhs_is_rowwise = lhs_contract_dim == lhs.ndim - 1
            rhs_is_rowwise = rhs_contract_dim == rhs.ndim - 1
            # Call JAX quantization so that XLA can do pattern matching (QDQ --> FP8 gemm)
            lhs_q = quantizer_set.x.quantize(
                lhs,
                is_rowwise=lhs_is_rowwise,
                is_colwise=not lhs_is_rowwise,
            )
            rhs_q = quantizer_set.kernel.quantize(
                rhs,
                is_rowwise=rhs_is_rowwise,
                is_colwise=not rhs_is_rowwise,
            )
            return _jax_gemm_fp8_impl(lhs_q, rhs_q)

    if (
        isinstance(lhs, jnp.ndarray)
        and isinstance(rhs, jnp.ndarray)
        and quantizer_set == noop_quantizer_set
    ):
        return jax.lax.dot_general(lhs, rhs, dim_nums, preferred_element_type=lhs.dtype)

    raise NotImplementedError("Not supporting multiplication of ScaledTensor and jnp.array")


def gemm(
    lhs: Union[jnp.ndarray, ScaledTensor],
    rhs: Union[jnp.ndarray, ScaledTensor],
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    quantizer_set: QuantizerSet = noop_quantizer_set,
) -> jnp.ndarray:
    """General matrix multiplication with optional quantization.

    Args:
        lhs: First input matrix.
        rhs: Second input matrix.
        contracting_dims: Tuple of two sequences representing the contracting dimensions.
            The first sequence represents the contracting dimensions of the first matrix,
            and the second sequence represents the contracting dimensions of the second matrix.
        quantizer_set: Set of quantizers for FP8 quantization of the output.
            If None, no quantization is applied and the output has the same dtype as the inputs.

    Returns:
        If quantizer_set is None:
            The matrix multiplication result.
            Shape: (M, N)
            Dtype: Same as input dtype
          If quantizer_set is provided:
            A ScaledTensor containing the quantized matrix multiplication result.
    """

    return _jax_gemm(lhs, rhs, contracting_dims, quantizer_set)


def swizzled_scale(scales):
    """Swizzle the scale tensor for FP8 GEMM"""
    assert scales.ndim == 2
    rows, cols = scales.shape
    scales = scales.reshape(rows // 128, 4, 32, cols // 4, 4)
    scales = jnp.transpose(scales, (0, 3, 2, 1, 4))
    scales = scales.reshape(rows, cols)
    return scales


def grouped_swizzled_scale(grouped_scale, group_sizes, original_shape, is_colwise, scaling_mode,
                           flatten_axis):
    """Swizzle the scale tensor for FP8 GEMM"""

    scale_shapes = scaling_mode.value.get_grouped_scale_shape(original_shape, group_sizes,
                                                              is_colwise, flatten_axis=flatten_axis)
    ptr = 0
    for scale_shape in scale_shapes:
        rows = math.prod(scale_shape[:flatten_axis])
        cols = math.prod(scale_shape[flatten_axis:])
        scale_size = rows * cols
        scale = grouped_scale[ptr: ptr + scale_size].reshape((rows, cols))
        if is_colwise:
            scale = jnp.transpose(scale, (1, 0))
            rows, cols = (cols, rows)
        scale = scale.reshape(rows // 128, 4, 32, cols // 4, 4)
        scale = jnp.transpose(scale, (0, 3, 2, 1, 4))
        grouped_scale = grouped_scale.at[ptr: ptr + scale_size].set(scale)
        ptr += scale_size
    return grouped_scale

# Note: already_transposed doesn't matter for the output shape
# x.shape = [B, D1, D2]
# contracting_dims = (2, )    --> output.shape = [1, B * D1, D2]
# contracting_dims = (0, 1, ) --> output.shape = [1, D2, B * D1]
# x.shape = [D1, D2]
# contracting_dims = (1, )    --> output.shape = [1, D1, D2]
# contracting_dims = (0, )    --> output.shape = [1, D2, D1]
# bm = lhs_remain_shape[0]
# bn = rhs_remain_shape[0]
# kl = lhs_3d.shape[-1]
# kr = rhs_3d.shape[-1]
# assert kl == kr, f"After shape normalization, contracting dim size mismatch: {kl} != {kr}"
# if (bm % 16 != 0) or (bn % 16 != 0) or (kl % 16 != 0):
#     print("grouped_gemm input pair {i} has invalid problem shape for lowering: ")
#     print(f"m = {bm}, n = {bn}, k = {kl}; ")
#     print("cuBLAS requires the problem shapes being multiples of 16")
#     assert (bm % 16 == 0) and (bn % 16 == 0) and (kl % 16 == 0)


def grouped_gemm(
        lhs: Union[jnp.ndarray, GroupedScaledTensor1x],
        rhs: Union[jnp.ndarray, GroupedScaledTensor1x],
        group_sizes: jnp.ndarray,
        contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (2,)),
        bias: jnp.ndarray = None,
        precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
        preferred_element_type: jnp.dtype = None,
        group_offset: jnp.array = None,
        quantizer_set: QuantizerSet = noop_quantizer_set,
        ) -> jnp.ndarray:
    """
    lhs: [M, K]
    rhs: [G, N, K]
    """
    # TODO(Phuong): implement the group_offset
    group_offset = group_offset or jnp.zeros((1,), jnp.int32)

    if isinstance(lhs, jnp.ndarray):
        assert isinstance(rhs, jnp.ndarray)
        out_dtype = lhs.dtype
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape
        lhs_data = lhs
        rhs_data = rhs
        lhs_scale_inv = rhs_scale_inv = jnp.empty((1,), jnp.float32)
        scaling_mode = ScalingMode.NO_SCALING
    elif isinstance(lhs, GroupedScaledTensor1x):
        assert isinstance(rhs, GroupedScaledTensor1x)
        out_dtype = lhs.dq_dtype
        lhs_shape = lhs.original_shape
        rhs_shape = rhs.original_shape
        lhs_data = lhs.data
        rhs_data = rhs.data
        lhs_scale_inv = lhs.scale_inv
        rhs_scale_inv = rhs.scale_inv
        assert lhs.scaling_mode == rhs.scaling_mode
        scaling_mode = lhs.scaling_mode
    else:
        raise TypeError("Unsupported lhs type object!")

    out_dtype = preferred_element_type or out_dtype

    lhs_contract_dim, rhs_contract_dim = contracting_dims

    lhs_is_trans = lhs_contract_dim[-1] != len(lhs_shape) - 1
    lhs_flatten_axis = len(lhs_contract_dim) * (1 if lhs_is_trans else -1)

    # rhs_shape [G, K, N]
    rhs_is_trans = rhs_contract_dim[0] != 1
    rhs_flatten_axis = -len(rhs_contract_dim) if rhs_is_trans else 1 + len(rhs_contract_dim)

    if not isinstance(lhs, ScaledTensor) and not isinstance(rhs, ScaledTensor) and quantizer_set != noop_quantizer_set:
        assert isinstance(quantizer_set.x, GroupedQuantizer)
        assert type(quantizer_set.x) is type(quantizer_set.kernel)
        lhs_q = quantizer_set.x.quantize(
                lhs,
                is_rowwise=not lhs_is_trans,
                is_colwise=lhs_is_trans,
                flatten_axis=lhs_flatten_axis,
                group_sizes=group_sizes,
                )
        rhs_q = quantizer_set.kernel.quantize(
                rhs,
                is_rowwise=rhs_is_trans,
                is_colwise=not rhs_is_trans,
                flatten_axis=rhs_flatten_axis,
                )
        lhs_data = lhs_q.data
        rhs_data = rhs_q.data
        lhs_scale_inv = lhs_q.scale_inv
        rhs_scale_inv = rhs_q.scale_inv

    assert not (lhs_data.dtype == jnp.float8_e5m2 and rhs_data.dtype == jnp.float8_e5m2
                ), "FP8 GEMM does not support E5M2 * E5M2"

    # Only support FP8 GEMM with NT layout on Hopper and other earlier GPUs
    # thus additional transpose is required
    if scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING and not is_gemm_with_all_layouts_supported():
        lhs_is_trans = False
        rhs_is_trans = True
        if lhs.data_layout == "T":
            lhs_contract_dim = tuple((lhs.data.ndim - 1 - i) % lhs.data.ndim for i in
                                     lhs_contract_dim)
        if rhs.data_layout == "T":
            rhs_contract_dim = tuple((rhs.data.ndim - 1 - i) % rhs.data.ndim for i in
                                     rhs_contract_dim)
        lhs_data = _shape_normalization(lhs.data, (lhs_contract_dim, ()), lhs.data_layout == "N")
        rhs_data = _shape_normalization(rhs.data, (rhs_contract_dim), rhs.data_layout == "T")

    if scaling_mode == ScalingMode.MXFP8_1D_SCALING:
        lhs_scale_inv = grouped_swizzled_scale(lhs.scale_inv, lhs.group_sizes, lhs_shape,
                                               lhs.is_colwise, lhs.scaling_mode, lhs.flatten_axis)
        rhs_scale_inv = grouped_swizzled_scale(rhs.scale_inv, rhs.group_sizes, rhs_shape,
                                               rhs.is_colwise, rhs.scaling_mode, rhs.flatten_axis)

    # Calling GroupedGEMM Custom Call
    K_lhs = math.prod(lhs_shape[i] for i in lhs_contract_dim)
    K_rhs = math.prod(rhs_shape[i] for i in rhs_contract_dim)
    assert K_lhs == K_rhs
    M = math.prod(_calculate_remaining_shape(lhs_shape, lhs_contract_dim))
    N = math.prod(_calculate_remaining_shape(rhs_shape, rhs_contract_dim)[1:])  # Exclude G
    assert group_sizes.size == rhs_shape[0]
    assert group_offset.size == 1

    has_bias = bias is not None
    assert not has_bias or bias.size == N
    bias = bias or jnp.empty((), jnp.float32)

    return GroupedGemmPrimitive.outer_primitive.bind(
        lhs_data,
        lhs_scale_inv,
        rhs_data,
        rhs_scale_inv,
        bias,
        group_sizes,
        group_offset,
        M=M,
        N=N,
        K=K_lhs,
        lhs_is_trans=lhs_is_trans,
        rhs_is_trans=rhs_is_trans,
        scaling_mode=scaling_mode.value,
        out_dtype=out_dtype,
        has_bias=has_bias,
    )
