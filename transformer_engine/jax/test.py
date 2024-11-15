# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX MLP modules"""

from typing import List, Tuple, Sequence, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .fp8 import FP8Helper, FP8MetaPackage

from .tensor import FP8_DQTensor, GeneralTensor


def activation_lu(input_tensor: Union[GeneralTensor, FP8_DQTensor],
                  activation_type: Sequence[Union[str, Callable]]):
    """
    Activation Unit
    """
    if len(activation_type) > 1:
        assert input_tensor.data.shape[-2] == 2  # Linear + GeLU
    output = _activation_lu(input_tensor, activation_type)
    return output


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _activation_lu(input_tensor: Union[GeneralTensor, FP8_DQTensor],
                  activation_type: Sequence[Union[str, Callable]]):

    _output, _ = _activation_lu_fwd_rule(input_tensor, activation_type)

    return _output


def _activation_lu_fwd_rule(input_tensor, activation_type):
    fwd_output = tex.general_act_lu(input_tensor, activation_type)
    return fwd_output, (input_tensor,)


def _activation_lu_bwd_rule(activation_type, ctx, g):
    (input_tensor,) = ctx
    return (input_tensor,)


_activation_lu.defvjp(_activation_lu_fwd_rule, _activation_lu_bwd_rule)


# rng = jax.random.PRNGKey(42)
# data_fp32 = jax.random.uniform(rng, (4, 4), jnp.float32, -10., 10.)
# scale = jnp.array(0.5)
# data_fp8 = FP8_DQTensor(data_fp32, scale)

# print("Original data: ", data_fp32)
# print("Quant/Dequant data: ", data_fp8.dequantize())
#
# gelu_jit = jax.jit(jax.nn.gelu)
# ref_out = gelu_jit(data_fp32)
# test_out = jax.tree.map(gelu_jit, data_fp8)
# print("Original GELU out: ", ref_out)
# print("FP8 GELU out: ", test_out.dequantize())

# Create test input
data = jnp.array([[1.0, -2.0], [-3.0, 4.0]])
x = GeneralTensor(data, dtype=jnp.float32)

# Test forward pass
result = activation_lu(x, "gelu")

assert isinstance(result, GeneralTensor)

expected = jnp.array([[1.0, 0.0], [0.0, 4.0]])
jnp.testing.assert_allclose(result.flatten(), expected)

# # Test gradient
# def wrapped_act_lu(x_data):
#     x_tensor = GeneralTensor(x_data, dtype=jnp.float32)
#     result = act_lu(x_tensor, activation_enum=0)
#     return result.dequantize()
#
# # Test gradient using grad
# grad_fn = jax.grad(wrapped_act_lu)
# grad_result = grad_fn(data)
# expected_grad = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # ReLU gradient
# jnp.testing.assert_allclose(grad_result, expected_grad)
#
# # Test with JIT
# jitted_act_lu = jax.jit(act_lu)
# jit_result = jitted_act_lu(x, activation_enum=0)
# assert isinstance(jit_result, GeneralTensor)
# jnp.testing.assert_allclose(jit_result.dequantize(), expected)
