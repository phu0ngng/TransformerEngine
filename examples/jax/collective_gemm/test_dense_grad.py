# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Comm+GEMM Overlap with TE/JAX"""
import os
import argparse
from functools import partial

from mpi4py import MPI

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

from common import assert_allclose
import transformer_engine.jax as te

# from transformer_engine.common import recipe
from transformer_engine.jax.sharding import (
    MeshResource,
    global_shard_guard,
)
from transformer_engine.jax.dense import dense
from transformer_engine.jax.cpp_extensions import (
    CommOverlapHelper,
    CommOverlapHelperSet,
)
import transformer_engine_jax as tex


jax.clear_caches()

# This script needs to be launched via `mpirun` with 1 process per GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
myrank = MPI.COMM_WORLD.Get_rank()
numranks = MPI.COMM_WORLD.Get_size()
jax.distributed.initialize(cluster_detection_method="mpi4py")

# TODO: do we really need this ?
assert (
    jax.local_device_count() == 1
), f"[{myrank}|{numranks}] Expected 1 GPU per process, found {jax.local_device_count()}"


parser = argparse.ArgumentParser()
parser.add_argument("--dp", type=int, default=1)
parser.add_argument("--fsdp", type=int, default=1)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--seq-length", type=int, default=8192)
parser.add_argument("--hidden-in", type=int, default=16384)
parser.add_argument("--hidden-out", type=int, default=53248)
parser.add_argument("--comm-type", type=str.upper, default="AG", choices=["AG", "RS"])
# parser.add_argument("--fp8-recipe", type=str.lower, default=None,
#                     choices=["fp8_currentscaling", "fp8_delayedscaling", "mxfp8"],
#                     )
# TODO: remove these two
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

assert (
    args.dp == 1 or args.fsdp == 1
), f"DP and FSDP should not be used at the same time! Got DP={args.dp} FSDP={args.fsdp}"
n_gpus = args.dp * args.fsdp * args.tp
assert n_gpus == numranks, f"We need {n_gpus} processes for {n_gpus} GPUs, got {numranks}!"

# Declare inputs
dtype = jnp.bfloat16
input_shape = (args.batch_size, args.seq_length, args.hidden_in)
kernel_shape = (args.hidden_in, args.hidden_out)
bias_shape = (args.hidden_out,)

rng = jax.random.PRNGKey(args.seed)
rng, params_rng = jax.random.split(rng)
params_rng, kernel_rng = jax.random.split(params_rng)
params_rng, bias_rng = jax.random.split(params_rng)
x = jax.random.normal(rng, input_shape, dtype=jnp.bfloat16)
kernel = jax.random.normal(kernel_rng, kernel_shape, dtype=jnp.bfloat16)
bias = jax.random.normal(bias_rng, bias_shape, dtype=jnp.bfloat16)

if myrank == 0:
    print(
        f"[{myrank}|{numranks}]\n"
        + f"  x:        {x.shape}\n"
        + f"  kernel: {kernel.shape}\n"
        + f"  bias:   {bias.shape}\n"
    )

fp8_recipe = None  # For now


# Single GPU evaluation
def eval_ref(x_, kernel_, bias_):
    output_ = dense(x_, kernel_, bias_, contracting_dims=((x_.ndim - 1,), (0,)))
    return jnp.mean(output_)


with te.fp8_autocast(enabled=fp8_recipe is not None, fp8_recipe=fp8_recipe):
    ref_output, ref_grads = jax.jit(jax.value_and_grad(eval_ref, range(3)))(x, kernel, bias)

# Device mesh and logical axis resources
FSDP_AXIS = "fsdp"
DP_AXIS = "data"
TP_AXIS = "tensor_sequence"

mesh_shape = {
    TP_AXIS: args.tp,
    DP_AXIS: args.dp,
    FSDP_AXIS: args.fsdp,
}

mesh_resource = MeshResource(
    dp_resource=None if args.dp == 1 else DP_AXIS,
    fsdp_resource=None if args.fsdp == 1 else FSDP_AXIS,
    tp_resource=None if args.tp == 1 else TP_AXIS,
)
devices = mesh_utils.create_device_mesh((n_gpus,), devices=jax.devices()[:n_gpus])
mesh = Mesh(np.array(devices).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))
if myrank == 0:
    print(f"[{myrank}|{numranks}] Device mesh: {mesh}\n")

if args.comm_type == "AG":
    input_specs = [(FSDP_AXIS, DP_AXIS), TP_AXIS, None]
    kernel_specs = [FSDP_AXIS, TP_AXIS]
    bias_specs = [
        TP_AXIS,
    ]
else:  # "RS"
    input_specs = [(FSDP_AXIS, DP_AXIS), None, TP_AXIS]
    kernel_specs = [TP_AXIS, FSDP_AXIS]
    bias_specs = [
        None,
    ]


# Multi GPU evaluation
def eval(x_, kernel_, bias_, comm_overlaps_):
    _output = dense(
        x_,
        kernel_,
        bias_,
        contracting_dims=((x_.ndim - 1,), (0,)),
        comm_overlaps=comm_overlaps_,
    )
    return jnp.mean(_output)


with (
    mesh,
    global_shard_guard(mesh_resource),
    te.fp8_autocast(
        enabled=fp8_recipe is not None,
        fp8_recipe=fp8_recipe,
        mesh_resource=mesh_resource,
    ),
):
    # Comm+GEMM overlap configs
    # TODO: make a CommOverlapHelperSet.create()
    buffer_shape = list(input_shape).copy()
    buffer_shape[0] = buffer_shape[0] // (args.dp * args.fsdp)
    fprop_1_overlap = CommOverlapHelper(
        comm_type=(tex.CommOverlapType.RS if args.comm_type == "RS" else tex.CommOverlapType.AG),
        method=tex.CommOverlapMethod.RING_EXCHANGE,
        buffer_shape=buffer_shape,
    )
    comm_overlaps = CommOverlapHelperSet(fprop=fprop_1_overlap)

    x_sharding = NamedSharding(mesh, PartitionSpec(*input_specs))
    kernel_sharding = NamedSharding(mesh, PartitionSpec(*kernel_specs))
    bias_sharding = NamedSharding(mesh, PartitionSpec(*bias_specs))
    x = jax.device_put(x, x_sharding)
    kernel = jax.device_put(kernel, kernel_sharding)
    bias = jax.device_put(bias, bias_sharding)

    input_shardings = (x_sharding, kernel_sharding, bias_sharding)
    output_shardings = (NamedSharding(mesh, PartitionSpec()), input_shardings)

    # TODO
    jitted_value_and_grad = jax.jit(
        jax.value_and_grad(eval, range(3)),
        static_argnums=(3,),
        in_shardings=input_shardings,
        out_shardings=output_shardings,
    )

    output, grads = jitted_value_and_grad(x, kernel, bias, comm_overlaps)

assertion_dtype = jnp.bfloat16
assert_allclose(output, ref_output, dtype=assertion_dtype)

labels = ("dX", "dKernel", "dBias")
for i, (ref, target) in enumerate(zip(ref_grads, grads)):
    if myrank == 0:
        print(
            f"[{myrank}|{numranks}] {labels[i]} : {target.shape}\n"
            + f"  Sharding: {target.sharding.spec}\n"
        )
    gathered = jax.lax.with_sharding_constraint(target, NamedSharding(mesh, PartitionSpec(None)))
    jax.block_until_ready(gathered)
    assert_allclose(ref, gathered, dtype=assertion_dtype)

tex.destroy_all_comm_overlap_buffers()
