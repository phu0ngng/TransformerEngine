# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX related extensions."""
import os
from pathlib import Path
from packaging import version

import setuptools

from .utils import get_cuda_include_dirs, all_files_in_dir, debug_build_enabled
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/JAX extensions."""
    return ["jax", "flax>=0.7.1"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/JAX extensions.

    Triton Package Selection:
        The triton package is selected based on NVTE_USE_PYTORCH_TRITON environment variable:

        Default (NVTE_USE_PYTORCH_TRITON unset or "0"):
            Returns 'triton' - OpenAI's standard package from PyPI.
            Install with: pip install triton

        NVTE_USE_PYTORCH_TRITON=1:
            Returns 'pytorch-triton' - for mixed JAX+PyTorch environments.
            Install with: pip install pytorch-triton --index-url https://download.pytorch.org/whl/cu121

            Note: Do NOT install pytorch-triton from PyPI directly - that's a placeholder.
    """
    use_pytorch_triton = bool(int(os.environ.get("NVTE_USE_PYTORCH_TRITON", "0")))

    triton_package = "pytorch-triton" if use_pytorch_triton else "triton"

    return [
        "numpy",
        triton_package,
    ]


def xla_path() -> str:
    """XLA root path lookup.
    Throws FileNotFoundError if XLA source is not found."""

    try:
        import jax

        if version.parse(jax.__version__) >= version.parse("0.5.0"):
            from jax import ffi  # pylint: disable=ungrouped-imports
        else:
            from jax.extend import ffi  # pylint: disable=ungrouped-imports

    except ImportError:
        if os.getenv("XLA_HOME"):
            xla_home = Path(os.getenv("XLA_HOME"))
        else:
            xla_home = "/opt/xla"
    else:
        xla_home = ffi.include_dir()

    if not os.path.isdir(xla_home):
        raise FileNotFoundError("Could not find xla source.")
    return xla_home


def setup_jax_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup PyBind11 extension for JAX support"""
    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = all_files_in_dir(extensions_dir, name_extension="cpp")

    # Header files
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
            xla_path(),
        ]
    )

    # Compile flags
    cxx_flags = ["-O3"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # NCCL EP (optional): when NVTE_WITH_NCCL_EP=1, define the macro so the
    # JAX extension's pybind.cpp + ep.cpp are guarded in. Headers come from
    # the same submodule the C++ side uses; lib/ is contributed by NCCL_EP_DIR.
    libraries = ["nccl"]
    if bool(int(os.getenv("NVTE_WITH_NCCL_EP", "0"))):
        # NCCL EP requires SM>=90 (Hopper+).
        archs_env = os.getenv("NVTE_CUDA_ARCHS", "")
        for a in archs_env.split(";"):
            a_num = "".join(c for c in a if c.isdigit())
            if a_num and int(a_num) < 90:
                raise RuntimeError(
                    "NVTE_WITH_NCCL_EP=1 requires CUDA arch >= 90 (Hopper or newer); "
                    f"got '{a}' in NVTE_CUDA_ARCHS."
                )
        cxx_flags.append("-DNVTE_WITH_NCCL_EP")
        nccl_ep_dir = os.getenv("NCCL_EP_DIR") or os.getenv("NCCL_EP_HOME")
        if not nccl_ep_dir:
            raise RuntimeError("NVTE_WITH_NCCL_EP=1 requires NCCL_EP_DIR or NCCL_EP_HOME env var")
        # In-tree submodule headers (preferred) + external fallback for nccl_ep.h.
        submod_inc = (
            common_header_files / ".." / "3rdparty" / "nccl" / "contrib" / "nccl_ep" / "include"
        ).resolve()
        if (submod_inc / "nccl_ep.h").exists():
            include_dirs.append(submod_inc)
        else:
            include_dirs.append(Path(nccl_ep_dir) / "include")
        libraries.append("nccl_ep")

    # Define TE/JAX as a Pybind11Extension
    from pybind11.setup_helpers import Pybind11Extension

    ext = Pybind11Extension(
        "transformer_engine_jax",
        sources=[str(path) for path in sources],
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args=cxx_flags,
        libraries=libraries,
    )
    if bool(int(os.getenv("NVTE_WITH_NCCL_EP", "0"))):
        nccl_ep_dir = os.getenv("NCCL_EP_DIR") or os.getenv("NCCL_EP_HOME")
        ext.library_dirs.append(str(Path(nccl_ep_dir) / "lib"))
        ext.runtime_library_dirs.append(str(Path(nccl_ep_dir) / "lib"))
        # Prefer submodule's nccl.h when present (matches the C++ side).
        submod_nccl_inc = (
            common_header_files / ".." / "3rdparty" / "nccl" / "build" / "include"
        ).resolve()
        if (submod_nccl_inc / "nccl.h").exists():
            ext.include_dirs.insert(0, str(submod_nccl_inc))
    return ext
