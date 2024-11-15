# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Tensor classes for TE/JAX
"""
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import core


class TensorAbstractValue(core.AbstractValue):
    """Define custom abstract class for our tensors"""
    def __init__(self, shape, dtype, tensor_type):
        self.shape = shape
        self.dtype = dtype
        self.tensor_type = tensor_type  # Either GeneralTensor or FP8Tensor

    def strip_weak_type(self):
        """Remove weak typing information while preserving the essential shape/dtype"""
        return self

    def join(self, other):
         """Combine two abstract values, used during shape/type inference"""
        if other is core.abstract_unit:
            return self
        if isinstance(other, TensorAbstractValue):
            if self.tensor_type != other.tensor_type:
                raise TypeError(f"Cannot join {self.tensor_type} with {other.tensor_type}")
            return TensorAbstractValue(
                self.shape,
                self.dtype,
                self.tensor_type
            )
        raise TypeError(f"Cannot join {self} with {other}")

    def update(self, **kwargs):
        """Create new abstract value with updated fields"""
        return TensorAbstractValue(
            kwargs.get('shape', self.shape),
            kwargs.get('dtype', self.dtype),
            kwargs.get('tensor_type', self.tensor_type)
        )

@register_pytree_node_class
class FP8MetaData:
    def __init__(self, scale=None, amax=None):
        self.scale = scale if scale is not None else jnp.ones((), dtype=jnp.float32)
        self.amax = amax if amax is not None else jnp.zeros((), dtype=jnp.float32)

    def tree_flatten(self):
        # scale and amax are dynamic data
        children = (self.scale, self.amax)
        # no static data
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        scale, amax = children
        return cls(scale, amax)

    def shape(self):
        """Return shapes of all dynamic data in order they appear in tree_flatten."""
        return (self.scale.shape, self.amax.shape)


class BaseTensor(ABC):
    def __init__(self):
        self.data = None
        self.dtype = None

    def tree_flatten(self):
        children = self.get_children()
        aux_data = self.get_aux_data()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    @abstractmethod
    def aval(self):
        pass

    @abstractmethod
    def get_children(self):
        pass

    def get_aux_data(self):
        return (self.dtype,)

    @abstractmethod
    def get_data_shape(self):
        return data.shape

@register_pytree_node_class
class FP8_DQTensor(BaseTensor):
    def __init__(self, data, dtype, scale = jnp.ones((), jnp.float32)):
        self.data = data
        self.dtype = dtype
        self.meta_scale = scale

    def get_children(self):
        return (self.data, self.meta_scale)

    def aval(self):
        return TensorAbstractValue(
            shape=(
                self.data.shape,      # fp8 data shape
                self.meta_scale.shape       # meta data shapes from FP8MetaData
            ),
            dtype=self.dtype, 
            tensor_type=type(self)
        )


@register_pytree_node_class
class FP8_QTensor(BaseTensor):
    def __init__(self, data, scale_inv, dtype, amax = jnp.zeros((), jnp.float32)):
        super().__init__()
        self.data = data
        self.dtype = dtype or jnp.float8_e4m3fn
        self.scale_inv = scale_inv
        self.meta_amax = jnp.zeros((), jnp.float32)

    def dequantize(self):
        return self.data.astype(self.dtype) * self.scale_inv

    @classmethod
    def quantize(cls, input, scale, q_dtype = jnp.float8_e4m3fn):
        compute_dtype = scale.dtype
        dtype_max = (jnp.finfo(q_dtype).max).astype(compute_dtype)
        scaled_x = x.astype(compute_dtype) * scale
        clipped_scaled_x = jnp.clip(scaled_x, -dtype_max, dtype_max).astype(q_dtype)
        scale_inv = 1.0 / scale
        return cls(clipped_scaled_x, scale_inv, q_dtype)

    def get_children(self):
        return (self.data, self.scale_inv, self.meta_amax)

    def aval(self):
        return TensorAbstractValue(
            shape=(
                self.data.shape,      # fp8 data shape
                self.scale_inv.shape, # scalar shape ()
                self.meta_amax,
            ),
            dtype=self.dtype, 
            tensor_type=type(self)
        )

@register_pytree_node_class
class GeneralTensor(BaseTensor):
    def __init__(self, data, dtype, meta : FP8MetaData = None):
        self.data = data
        self.dtype = dtype

    def get_children(self):
        return (self.data)

    def aval(self):
        return TensorAbstractValue(
            shape= (self.data.shape),      
            dtype=self.dtype, 
            tensor_type=type(self)
        )

