# Project-specific instructions

## Sandbox

- `nvidia-smi` and any GPU access require `dangerouslyDisableSandbox: true` on Bash calls. The sandbox blocks the NVIDIA driver socket and `nvidia-smi` reports "couldn't communicate with the NVIDIA driver". Run with sandbox disabled to see GPUs and run CUDA/NCCL workloads.

## TE build

- Always specify `NVTE_CUDA_ARCHS` for the target SM, **and include a base arch ≤ 90** (the TE CMake removes `100/101/110/120` from `CMAKE_CUDA_ARCHITECTURES` and reroutes them into family-specific lists; if nothing remains the target's `CUDA_ARCHITECTURES` becomes empty and configure errors with "CUDA_ARCHITECTURES is empty for target"). Use `"90;100"` for B300/sm_103, `"90"` for H100, etc. Single bare SMs like `103` configure cmake with `compute_103` (generic) and fail later with "Compiled for the generic architecture, while utilizing arch-specific features" in `ptx.cuh` — the family-specific suffix (`a`/`f`) is only added inside the TE CMake when the major-family arch (100/120) is passed.
- Set `NVTE_BUILD_THREADS_PER_JOB` (default 2) to speed up nvcc — try `8` on a fat box, `4` if memory is tight. Combine with `--no-build-isolation` so pip reuses the system toolchain.
- Standard fast-build invocation (B300, JAX framework, NCCL EP always on):
  ```bash
  NVTE_CUDA_ARCHS="90;100" NVTE_FRAMEWORK=jax NVTE_BUILD_THREADS_PER_JOB=8 \
    pip install --no-build-isolation -e .
  ```
- NCCL EP requires system NCCL >= 2.30.4. `setup.py` always builds `libnccl_ep.so` from `3rdparty/nccl/contrib/nccl_ep`; delete `3rdparty/nccl/build/lib/libnccl_ep.so` to force a rebuild after editing `contrib/nccl_ep` sources.
