# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for TE einsum operation with FP8 quantization."""

import jax
import jax.numpy as jnp
import pytest
from jax import value_and_grad

from utils import assert_allclose, pytest_parametrize_wrapper
from transformer_engine.jax.einsum import einsum
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    QuantizeMeta,
    QuantizeMetaSet,
)
from transformer_engine.jax.quantize import helper


# Test parameters
DTYPES = [jnp.bfloat16, jnp.float32]
SIMPLE_MATMUL_CASES = [
    (128, 256, 512),
    (64, 128, 256),
    (256, 512, 1024),
]
BATCHED_MATMUL_CASES = [
    (4, 128, 256, 512),
    (8, 64, 128, 256),
]
MOE_CASES = [
    # (B, S, M, E, C, H)
    # B: Batch size
    # S: Sequence length (number of tokens)
    # M: Model dimension (hidden size)
    # E: Number of experts
    # C: Capacity (max tokens per expert)
    # H: Hidden dimension (MLP intermediate size)
    (2, 8, 128, 8, 2, 512),
    (4, 16, 64, 4, 4, 256),
]

# Get supported scaling modes and recipes
supported_scaling_modes = helper.get_supported_scaling_modes()
supported_recipes = helper.get_supported_quantization_recipes()
supported_recipes = [pytest.param(r, id=r.__class__.__name__) for r in supported_recipes]


@pytest.fixture(autouse=True, scope="module")
def init():
    """WAR for CUDA uninitialize error"""
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


class TestEinsumBasic:
    """Test basic einsum operations."""
    
    @pytest_parametrize_wrapper("M,K,N", SIMPLE_MATMUL_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_simple_matmul(self, M, K, N, dtype):
        """Test simple matrix multiplication: ij,jk->ik"""
        A = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
        B = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)
        
        # Without FP8
        result = einsum("ij,jk->ik", A, B)
        expected = jnp.einsum("ij,jk->ik", A, B)
        
        assert result.shape == (M, N)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper("B,M,K,N", BATCHED_MATMUL_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_batched_matmul(self, B, M, K, N, dtype):
        """Test batched matrix multiplication: bij,bjk->bik"""
        A = jax.random.normal(jax.random.PRNGKey(0), (B, M, K), dtype=dtype)
        B_mat = jax.random.normal(jax.random.PRNGKey(1), (B, K, N), dtype=dtype)
        
        result = einsum("bij,bjk->bik", A, B_mat)
        expected = jnp.einsum("bij,bjk->bik", A, B_mat)
        
        assert result.shape == (B, M, N)
        assert_allclose(result, expected, dtype=dtype)


class TestEinsumMoE:
    """Test MoE-specific einsum operations."""
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_dispatch(self, B, S, M, E, C, H, dtype):
        """Test MoE dispatch: BSM,BSEC->EBCM"""
        tokens = jax.random.normal(jax.random.PRNGKey(0), (B, S, M), dtype=dtype)
        routing = jax.random.normal(jax.random.PRNGKey(1), (B, S, E, C), dtype=dtype)
        
        result = einsum("BSM,BSEC->EBCM", tokens, routing)
        expected = jnp.einsum("BSM,BSEC->EBCM", tokens, routing)
        
        assert result.shape == (E, B, C, M)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_mlp_up(self, B, S, M, E, C, H, dtype):
        """Test MoE MLP up projection: EBCM,EMH->EBCH"""
        dispatched = jax.random.normal(jax.random.PRNGKey(0), (E, B, C, M), dtype=dtype)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, M, H), dtype=dtype)
        
        result = einsum("EBCM,EMH->EBCH", dispatched, weights)
        expected = jnp.einsum("EBCM,EMH->EBCH", dispatched, weights)
        
        assert result.shape == (E, B, C, H)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_mlp_down(self, B, S, M, E, C, H, dtype):
        """Test MoE MLP down projection: EBCH,EHM->EBCM"""
        hidden = jax.random.normal(jax.random.PRNGKey(0), (E, B, C, H), dtype=dtype)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, H, M), dtype=dtype)
        
        result = einsum("EBCH,EHM->EBCM", hidden, weights)
        expected = jnp.einsum("EBCH,EHM->EBCM", hidden, weights)
        
        assert result.shape == (E, B, C, M)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_output(self, B, S, M, E, C, H, dtype):
        """Test MoE output combination: EBCM,BSEC->BSM"""
        expert_outputs = jax.random.normal(jax.random.PRNGKey(0), (E, B, C, M), dtype=dtype)
        routing = jax.random.normal(jax.random.PRNGKey(1), (B, S, E, C), dtype=dtype)
        
        result = einsum("EBCM,BSEC->BSM", expert_outputs, routing)
        expected = jnp.einsum("EBCM,BSEC->BSM", expert_outputs, routing)
        
        assert result.shape == (B, S, M)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_complete_forward(self, B, S, M, E, C, H, dtype):
        """Test complete MoE forward pass with all four einsum operations."""
        # Inputs
        tokens = jax.random.normal(jax.random.PRNGKey(0), (B, S, M), dtype=dtype)
        routing = jax.random.normal(jax.random.PRNGKey(1), (B, S, E, C), dtype=dtype)
        up_weights = jax.random.normal(jax.random.PRNGKey(2), (E, M, H), dtype=dtype)
        down_weights = jax.random.normal(jax.random.PRNGKey(3), (E, H, M), dtype=dtype)
        
        # 1. Dispatch: BSM,BSEC -> EBCM
        dispatched = einsum("BSM,BSEC->EBCM", tokens, routing)
        assert dispatched.shape == (E, B, C, M)
        
        # 2. MLP Up: EBCM,EMH -> EBCH
        hidden = einsum("EBCM,EMH->EBCH", dispatched, up_weights)
        assert hidden.shape == (E, B, C, H)
        
        # 3. MLP Down: EBCH,EHM -> EBCM
        expert_out = einsum("EBCH,EHM->EBCM", hidden, down_weights)
        assert expert_out.shape == (E, B, C, M)
        
        # 4. Output: EBCM,BSEC -> BSM
        output = einsum("EBCM,BSEC->BSM", expert_out, routing)
        assert output.shape == (B, S, M)


class TestEinsumAutodiff:
    """Test automatic differentiation through einsum."""
    
    @pytest_parametrize_wrapper("M,K,N", SIMPLE_MATMUL_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_simple_grad(self, M, K, N, dtype):
        """Test gradients for simple matrix multiplication."""
        A = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=dtype)
        B = jax.random.normal(jax.random.PRNGKey(1), (K, N), dtype=dtype)
        
        def loss_fn(a, b):
            result = einsum("ij,jk->ik", a, b)
            return jnp.sum(result ** 2)
        
        # Compute gradients
        loss, grads = value_and_grad(loss_fn, argnums=(0, 1))(A, B)
        
        assert grads[0].shape == A.shape
        assert grads[1].shape == B.shape
        assert not jnp.isnan(loss)
    
    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    def test_moe_complete_grad(self, B, S, M, E, C, H, dtype):
        """Test gradients through complete MoE pipeline."""
        def moe_forward(tokens, routing, up_w, down_w):
            # Complete MoE forward pass
            dispatched = einsum("BSM,BSEC->EBCM", tokens, routing)
            hidden = einsum("EBCM,EMH->EBCH", dispatched, up_w)
            expert_out = einsum("EBCH,EHM->EBCM", hidden, down_w)
            output = einsum("EBCM,BSEC->BSM", expert_out, routing)
            return jnp.sum(output ** 2)
        
        tokens = jax.random.normal(jax.random.PRNGKey(0), (B, S, M), dtype=dtype)
        routing = jax.random.normal(jax.random.PRNGKey(1), (B, S, E, C), dtype=dtype)
        up_weights = jax.random.normal(jax.random.PRNGKey(2), (E, M, H), dtype=dtype)
        down_weights = jax.random.normal(jax.random.PRNGKey(3), (E, H, M), dtype=dtype)
        
        # Compute gradients
        loss, grads = value_and_grad(moe_forward, argnums=(0, 1, 2, 3))(
            tokens, routing, up_weights, down_weights
        )
        
        # Check gradient shapes
        assert grads[0].shape == tokens.shape
        assert grads[1].shape == routing.shape
        assert grads[2].shape == up_weights.shape
        assert grads[3].shape == down_weights.shape
        assert not jnp.isnan(loss)


class TestEinsumPerExpertQuantizers:
    """Test einsum with per-expert quantizers using different FP8 recipes."""
    
    @pytest_parametrize_wrapper("recipe", supported_recipes)
    @pytest_parametrize_wrapper("dtype", [jnp.bfloat16])
    def test_per_expert_quantizers_different_recipes(self, recipe, dtype):
        """Test einsum with per-expert quantizers using different FP8 recipes."""
        B, S, M, E, C, H = 2, 4, 64, 4, 2, 128
        
        # Create per-expert quantizers with different recipes
        quantizer_sets = [
            QuantizerFactory.create_set(
                fp8_recipe=recipe,
                quantize_meta_set=QuantizeMetaSet(
                    x=QuantizeMeta(),
                    kernel=QuantizeMeta(),
                    grad=QuantizeMeta()
                )
            ) for _ in range(E)
        ]
        
        dispatched = jax.random.normal(jax.random.PRNGKey(0), (E, B, C, M), dtype=dtype)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, M, H), dtype=dtype)
        
        # Test with FP8 quantization
        result = einsum("EBCM,EMH->EBCH", dispatched, weights, 
                       quantizer_sets=quantizer_sets)
        expected = jnp.einsum("EBCM,EMH->EBCH", dispatched, weights)
        
        assert result.shape == (E, B, C, H)
        assert_allclose(result, expected, dtype=dtype)
    
    @pytest_parametrize_wrapper(
        "recipe", supported_recipes,
        "dtype", [jnp.bfloat16],
    )
    def test_per_expert_quantizers_with_grad(self, recipe, dtype):
        """Test gradients work with per-expert FP8 quantizers."""
        B, S, M, E, C, H = 2, 4, 64, 4, 2, 128
        
        # Create per-expert quantizers
        quantizer_sets = [
            QuantizerFactory.create_set(
                fp8_recipe=recipe,
                quantize_meta_set=QuantizeMetaSet(
                    x=QuantizeMeta(),
                    kernel=QuantizeMeta(),
                    grad=QuantizeMeta()
                )
            ) for _ in range(E)
        ]
        
        def loss_fn(x, w):
            result = einsum("EBCM,EMH->EBCH", x, w, quantizer_sets=quantizer_sets)
            return jnp.sum(result ** 2)
        
        dispatched = jax.random.normal(jax.random.PRNGKey(0), (E, B, C, M), dtype=dtype)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, M, H), dtype=dtype)
        
        # Compute gradients with FP8
        loss, grads = value_and_grad(loss_fn, argnums=(0, 1))(dispatched, weights)
        
        # Check gradient shapes
        assert grads[0].shape == dispatched.shape
        assert grads[1].shape == weights.shape
        assert not jnp.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
