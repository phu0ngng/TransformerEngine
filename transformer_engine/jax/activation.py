# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX Activation modules"""

from typing import List, Tuple, Sequence, Union, Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex

from .tensor import (
    Float8ScaledQuantizer,
    ScaledTensor,
    Float8ScaledTensor,
)


def activation_lu(x: jnp.ndarray, activation_type: Sequence[Union[str, Callable]], quantizer :
                  Optional[Float8ScaledQuantizer]) -> Union[jnp.ndarray, Float8ScaledTensor]:
    """
    Activation Unit
    """
    if len(activation_type) > 1:
        assert x.shape[-2] == 2  # Linear + GeLU
    output = _activation_lu(x, activation_type, quantizer)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _activation_lu(x, activation_type, quantizer):
    _output, _ = _activation_lu_fwd_rule(x, activation_type, quantizer)
    return _output


def _activation_lu_fwd_rule(x, activation_type, quantizer):
    fwd_output = tex.general_act_lu(x, activation_type, quantizer)
    return fwd_output, (x, quantizer)


def _activation_lu_bwd_rule(activation_type, ctx, g):
    (x, _) = ctx
    if isinstance(g, ScaledTensor):
        g = g.dequantize()
    assert x.dtype == g.dtype
    dx = tex.dact_lu(g, x, activation_type)
    dx = jnp.reshape(dx, x.shape)
    return (dx, None)


_activation_lu.defvjp(_activation_lu_fwd_rule, _activation_lu_bwd_rule)
