# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Einsum operation with FP8 quantization support for Transformer Engine in JAX.

This module provides an einsum implementation that decomposes einsum operations into
a sequence of GEMMs, each with its own quantizer for FP8 support. It follows the
pattern of jax.numpy.einsum but uses TE's optimized GEMM operations.

This module provides a flexible einsum implementation optimized for Mixture-of-Experts (MoE)
models with quantization support. It leverages JAX's vmap and TE's dense layer to
efficiently handle complex multi-dimensional tensor contractions.

Key Features:
    - **Per-expert quantization**: Each expert can have independent scaling and quantization parameters
    - **Automatic differentiation**: Full gradient support via dense layer's VJP
    - **Efficient batching**: Uses vmap to vectorize over batch/expert dimensions
    - **Flexible API**: Supports both shared and per-expert quantizers

Example - MoE Forward Pass with Per-Expert FP8:
    ```python
    from transformer_engine.jax.einsum import einsum
    from transformer_engine.jax.quantize import QuantizerFactory, QuantizeMeta, QuantizeMetaSet

    # Create per-expert quantizers (E experts)
    quantizer_sets = [
        QuantizerFactory.create_set(
            fp8_recipe=recipe,
            quantize_meta_set=QuantizeMetaSet(
                x=QuantizeMeta(), kernel=QuantizeMeta(), grad=QuantizeMeta()
            )
        ) for _ in range(num_experts)
    ]

    # MoE pipeline with per-expert quantization
    # 1. Dispatch: BSM,BSEC -> EBCM (no quantization - routing operation)
    dispatched = einsum("BSM,BSEC->EBCM", tokens, routing)
    
    # 2. MLP Up: EBCM,EMH -> EBCH (per-expert quantization)
    hidden = einsum("EBCM,EMH->EBCH", dispatched, expert_up_weights,
                   quantizer_sets=expert_quantizers, quantizer_dim='E')
    
    # 3. MLP Down: EBCH,EHM -> EBCM (per-expert quantization)
    expert_out = einsum("EBCH,EHM->EBCM", hidden, expert_down_weights,
                       quantizer_sets=expert_quantizers, quantizer_dim='E')
    
    # 4. Output: EBCM,BSEC -> BSM (no quantization - routing operation)
    output = einsum("EBCM,BSEC->BSM", expert_out, routing)
    ```

Implementation Details:
    The einsum function works by:
    1. Parsing the einsum equation to identify batch and contracting dimensions
    2. Creating a vmapped version of TE's dense layer
    3. Vmapping over both batch dimensions AND quantizer_sets (for per-expert quantization)
    4. Leveraging dense's existing VJP for automatic differentiation

    This design reuses TE's well-tested dense layer infrastructure while enabling
    flexible per-expert quantization for MoE models.
"""

from typing import Tuple, Sequence, Optional, List
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from .dense import dense
from .quantize import (
    QuantizerSet,
    noop_quantizer_set,
)


def _parse_einsum_input(equation: str, *operands) -> Tuple[str, List[str], str]:
    """Parse einsum equation into input specs and output spec.

    Args:
        equation: Einsum equation string (e.g., "ij,jk->ik" or "BNSM,BNSEC->EBNCM")
        operands: Input tensors

    Returns:
        Tuple of (equation, input_specs, output_spec)
    """
    # Remove spaces
    equation = equation.replace(' ', '')

    if '->' in equation:
        inputs_str, output_str = equation.split('->')
        input_specs = inputs_str.split(',')
    else:
        # Implicit output mode
        inputs_str = equation
        input_specs = inputs_str.split(',')
        # Compute implicit output
        all_indices = set()
        for spec in input_specs:
            all_indices.update(spec)
        output_str = ''.join(sorted(all_indices))

    return equation, input_specs, output_str


def _find_contracting_and_batch_dims(lhs_spec: str, rhs_spec: str, output_spec: str):
    """Find contracting and batch dimensions for a GEMM operation.

    Args:
        lhs_spec: Index specification for LHS (e.g., "BNSM")
        rhs_spec: Index specification for RHS (e.g., "BNSEC")
        output_spec: Index specification for output (e.g., "EBNCM")

    Returns:
        Tuple of (lhs_contracting, rhs_contracting, lhs_batch, rhs_batch)
    """
    # Contracting dimensions: indices in both lhs and rhs but not in output
    lhs_set = set(lhs_spec)
    rhs_set = set(rhs_spec)
    output_set = set(output_spec)

    contracting_indices = (lhs_set & rhs_set) - output_set

    # Batch dimensions: indices in lhs, rhs, and output
    batch_indices = lhs_set & rhs_set & output_set

    # Find positions
    lhs_contracting = tuple(i for i, c in enumerate(lhs_spec) if c in contracting_indices)
    rhs_contracting = tuple(i for i, c in enumerate(rhs_spec) if c in contracting_indices)
    lhs_batch = tuple(i for i, c in enumerate(lhs_spec) if c in batch_indices)
    rhs_batch = tuple(i for i, c in enumerate(rhs_spec) if c in batch_indices)

    return lhs_contracting, rhs_contracting, lhs_batch, rhs_batch


def _einsum_to_gemm_info(equation: str, *operands):
    """Extract GEMM information from einsum equation.

    Args:
        equation: Einsum equation
        operands: Input tensors

    Returns:
        Dict with keys: lhs_idx, rhs_idx, contracting_dims, batch_dims, output_spec
    """
    equation, input_specs, output_spec = _parse_einsum_input(equation, *operands)

    if len(input_specs) != 2:
        raise NotImplementedError(f"Einsum with {len(input_specs)} operands not yet supported")

    lhs_spec, rhs_spec = input_specs

    lhs_contracting, rhs_contracting, lhs_batch, rhs_batch = _find_contracting_and_batch_dims(
        lhs_spec, rhs_spec, output_spec
    )

    return {
        'lhs_idx': 0,
        'rhs_idx': 1,
        'lhs_spec': lhs_spec,
        'rhs_spec': rhs_spec,
        'output_spec': output_spec,
        'contracting_dims': (lhs_contracting, rhs_contracting),
        'batch_dims': (lhs_batch, rhs_batch),
    }


def einsum(
    equation: str,
    *operands: jnp.ndarray,
    quantizer_sets: Optional[List[QuantizerSet]] = None,
    quantizer_dim: Optional[str] = None,
    operand_axes: Optional[List[Tuple[str, ...]]] = None,
    output_axes: Optional[Tuple[str, ...]] = None,
) -> jnp.ndarray:
    """Perform einsum operation with optional FP8 quantization using vmap + dense.

    This function implements einsum by:
    1. Identifying batch dimensions
    2. Using vmap to vectorize over batch dimensions
    3. Calling the existing dense() function which has VJP already implemented

    Each batched GEMM can have its own quantizer_set, enabling per-expert
    quantization in MoE models.

    Args:
        equation: Einsum equation string (e.g., "ij,jk->ik", "BSM,BSEC->EBCM")
        *operands: Input tensors
        quantizer_sets: List or tuple of QuantizerSets. Length must match the size of
                       the dimension specified by quantizer_dim. If None, creates noop quantizers.
        quantizer_dim: Index label indicating which dimension the quantizers correspond to.
                      For MoE, this is typically 'E' (expert dimension). If None and
                      quantizer_sets is provided, assumes first batch dimension at position 0.
        operand_axes: List of logical axes tuples for sharding each operand
        output_axes: Logical axes for sharding the output

    Returns:
        Result of the einsum operation

    Examples:
        # Simple matrix multiplication with FP8
        result = einsum("ij,jk->ik", A, B, quantizer_sets=my_quantizer_set)

        # MoE with per-expert quantizers (E experts)
        expert_quantizers = [quantizer_e0, quantizer_e1, ..., quantizer_eN]
        result = einsum("EBNCM,EMH->EBNCH", tokens, weights,
                       quantizer_sets=expert_quantizers)
    """
    if operand_axes is None:
        operand_axes = [None] * len(operands)
    
    if len(operands) != 2:
        raise NotImplementedError("Only 2-operand einsum currently supported")
    
    # Parse einsum to get GEMM info
    gemm_info = _einsum_to_gemm_info(equation, *operands)
    contracting_dims = gemm_info['contracting_dims']
    batch_dims = gemm_info['batch_dims']
    lhs_spec = gemm_info['lhs_spec']
    rhs_spec = gemm_info['rhs_spec']
    
    lhs, rhs = operands
    
    # Find quantizer dimension
    quantizer_dim_lhs = None
    quantizer_dim_rhs = None
    
    if quantizer_dim is not None:
        # Find position of quantizer_dim in lhs and rhs specs
        if quantizer_dim in lhs_spec:
            quantizer_dim_lhs = lhs_spec.index(quantizer_dim)
        if quantizer_dim in rhs_spec:
            quantizer_dim_rhs = rhs_spec.index(quantizer_dim)
        
        if quantizer_dim_lhs is None and quantizer_dim_rhs is None:
            raise ValueError(
                f"quantizer_dim '{quantizer_dim}' not found in equation '{equation}'"
            )
    else:
        # Default: use first batch dimension at position 0 (if exists)
        if batch_dims[0] and 0 in batch_dims[0]:
            quantizer_dim_lhs = 0
        if batch_dims[1] and 0 in batch_dims[1]:
            quantizer_dim_rhs = 0
    
    has_quantizer_dim = quantizer_dim_lhs is not None or quantizer_dim_rhs is not None
    
    # Determine expected quantizer_sets length
    if has_quantizer_dim:
        if quantizer_dim_lhs is not None:
            expected_length = lhs.shape[quantizer_dim_lhs]
        else:
            expected_length = rhs.shape[quantizer_dim_rhs]
    else:
        expected_length = 1
    
    # Validate quantizer_sets
    if quantizer_sets is None:
        quantizer_sets = [noop_quantizer_set] * expected_length
    elif not isinstance(quantizer_sets, (list, tuple)):
        raise TypeError(f"quantizer_sets must be a list or tuple, got {type(quantizer_sets)}")
    elif len(quantizer_sets) != expected_length:
        raise ValueError(
            f"quantizer_sets length ({len(quantizer_sets)}) must match "
            f"dimension {quantizer_dim if quantizer_dim else '(first batch dim)'} "
            f"size ({expected_length})"
        )
    
    # Validate that this is NN layout (required by dense)
    # For NN: lhs last dim must contract, rhs last dim must NOT contract
    lhs_ndim = len(gemm_info['lhs_spec'])
    rhs_ndim = len(gemm_info['rhs_spec'])
    lhs_last_contracts = (lhs_ndim - 1) in contracting_dims[0]
    rhs_last_contracts = (rhs_ndim - 1) in contracting_dims[1]
    
    if not lhs_last_contracts or rhs_last_contracts:
        raise ValueError(
            f"Einsum equation '{equation}' does not correspond to NN layout. "
            f"TE dense requires last dimension of LHS to contract and last dimension of RHS to not contract. "
            f"Got lhs_contracting={contracting_dims[0]}, rhs_contracting={contracting_dims[1]}"
        )

    # Create vmapped dense function for batch dimensions
    has_batch_dims = bool(batch_dims[0] or batch_dims[1])
    
    if has_batch_dims:
        # Adjust contracting dims after removing batch dimensions
        # When we vmap, batch dimensions are removed, so we need to adjust indices
        lhs_batch_sorted = sorted(batch_dims[0])
        rhs_batch_sorted = sorted(batch_dims[1])
        
        # Compute adjusted contracting dims (after batch dims are removed by vmap)
        adj_lhs_contracting = tuple(
            dim - sum(1 for b in lhs_batch_sorted if b < dim)
            for dim in contracting_dims[0]
        )
        adj_rhs_contracting = tuple(
            dim - sum(1 for b in rhs_batch_sorted if b < dim)
            for dim in contracting_dims[1]
        )
        adj_contracting_dims = (adj_lhs_contracting, adj_rhs_contracting)
        
        # Create dense wrapper function
        def dense_with_quantizer(lhs_single, rhs_single, quantizer_set):
            """Dense with explicit quantizer argument for vmapping."""
            return dense(
                lhs_single, rhs_single, None,
                contracting_dims=adj_contracting_dims,
                transpose_batch_sequence=False,
                input_axes=operand_axes[0],
                kernel_axes=operand_axes[1],
                output_axes=output_axes,
                quantizer_set=quantizer_set,
            )

        # Apply vmap for all batch dimensions
        vmapped_func = dense_with_quantizer
        for idx, (lhs_dim, rhs_dim) in enumerate(zip(lhs_batch_sorted, rhs_batch_sorted)):
            adj_lhs = lhs_dim - idx
            adj_rhs = rhs_dim - idx
            adj_quantizer = 0  # Quantizer array always at position 0 after each vmap, when no quantizer dimension it does not matter
            
            vmapped_func = jax.vmap(
                vmapped_func,
                in_axes=(adj_lhs, adj_rhs, adj_quantizer),
                out_axes=0
            )
        
        output = vmapped_func(lhs, rhs, quantizer_sets)
    else:
        # No batch dimensions - direct dense call
        # quantizer_set length already validated to be 1
        output = dense(
            lhs, rhs, None,
            contracting_dims=contracting_dims,
            transpose_batch_sequence=False,
            input_axes=operand_axes[0],
            kernel_axes=operand_axes[1],
            output_axes=output_axes,
            quantizer_set=quantizer_sets[0],
        )
    
    return output
