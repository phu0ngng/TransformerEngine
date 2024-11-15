# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tensor classes for TE/JAX
"""
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .fp8 import FP8Helper, AmaxComputeAlgo


@register_pytree_node_class
class PyTree(ABC):

    def tree_flatten(self):
        children = self.get_children()
        aux_data = self.get_aux_data()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    @abstractmethod
    def get_children(self):
        pass

    @abstractmethod
    def get_aux_data(self):
        pass

@register_pytree_node_class
class BaseQuantizer(PyTree):
    def __init__(self, dtype):
        self.dtype = dtype

    @classmethod
    @abstractmethod
    def create(cls, dtype):
        pass

@register_pytree_node_class
class Float8ScaledQuantizer(BaseQuantizer):
    def __init__(self, scale: jnp.ndarray, amax_history: jnp.ndarray, dtype):
        self.scale = scale
        self.amax_history = amax_history
        self.dtype = dtype

    def get_children(self):
        return (self.scale, self.amax_history)

    def get_aux_data(self):
        return (self.dtype,)

    @classmethod
    def create(cls, dtype):
        assert dtype in [jnp.float8_e5m2, jnp.float8_e4m3fn]
        scale = jnp.ones((1,), jnp.float32)
        amax_history = jnp.zeros((FP8Helper.AMAX_HISTORY_LEN,), jnp.float32)
        return cls(scale, amax_history, dtype)

#    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def update(self, new_amax: jnp.ndarray):
        """
        Update amax_history with new_amax and calculate the scale
        """
        # 1. Update amax_history[0] = new_amax
        self.amax_history = self.amax_history.at[0].set(new_amax[0])

        # 2. Calculate the current scale
        fp8_max = jnp.astype(jnp.finfo(self.dtype).max, jnp.float32)

        if FP8Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
            amax = jnp.max(self.amax_history, axis=-1, keepdims=True)
        else:
            amax = self.amax_history[0:1]

        sf = (fp8_max / amax) / (2**FP8Helper.MARGIN)
        sf = jnp.where(amax > 0.0, sf, self.scale)
        sf = jnp.where(jnp.isfinite(amax), sf, self.scale)
        self.scale = self.scale.at[0].set(sf[0])

        # 3. Roll and reset amax_history[0] = 0.
        updated_amax = jnp.roll(self.amax_history, -1, -1)
        self.amax_history = updated_amax.at[0].set(0.)

@register_pytree_node_class
class ScaledTensor(PyTree):
    def __init__(self, data, scale):
        self.data = data
        self.scale = scale

    @abstractmethod
    def dequantize(self):
        pass

    @classmethod
    @abstractmethod
    def quantize(cls, data):
        pass

    def get_children(self):
        return (self.data, self.scale)

    def get_aux_data(self):
        return ()

@register_pytree_node_class
class Float8ScaledTensor(ScaledTensor):
    def __init__(self, data, scale):
        super().__init__(data, scale)

    def dequantize(self, dqtype = jnp.bfloat16):
        return jnp.asarray(self.data.astype(self.scale.dtype) * self.scale, dqtype)

    @classmethod
    def quantize(cls, data, quantizer: Float8ScaledQuantizer):
        dtype = quantizer.dtype
        scale = quantizer.scale
        compute_dtype = scale.dtype
        dtype_max = (jnp.finfo(dtype).max).astype(compute_dtype)
        scaled_x = data.astype(compute_dtype) * scale
        clipped_scaled_x = jnp.clip(scaled_x, -dtype_max, dtype_max).astype(dtype)
        scale_inv = 1.0 / scale
        return cls(clipped_scaled_x, scale_inv)
