"""Loader of the CUDA decision kernels (``csrc/vkv_decide.cu``): built on first use with
``torch.utils.cpp_extension.load`` (cached under the torch extensions directory) by the CUDA
toolkit matching torch -- ``CUDA_HOME`` if set, else the pip ``nvidia/cu*`` toolkit next to
torch -- without fast math and without FMA contraction (the reference's op-by-op rounding)."""
from __future__ import annotations

import glob
import os
import sys

import torch

_EXT = None
_SRC = os.path.join(os.path.dirname(__file__), "csrc", "vkv_decide.cu")


def _toolkit() -> str | None:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    site = os.path.dirname(os.path.dirname(torch.__file__))
    for d in sorted(glob.glob(os.path.join(site, "nvidia", "cu*")), reverse=True):
        if os.path.exists(os.path.join(d, "bin", "nvcc")):
            return d
    return None


def ext():
    global _EXT
    if _EXT is None:
        import torch.utils.cpp_extension as ce
        home = _toolkit()
        if home is not None:
            os.environ["CUDA_HOME"] = home
            ce.CUDA_HOME = home
            os.environ["PATH"] = os.path.join(home, "bin") + os.pathsep + os.environ.get("PATH", "")
        # ninja lives next to the interpreter in a venv
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
        major, minor = torch.cuda.get_device_capability()
        _EXT = ce.load(name="vkv_decide", sources=[_SRC], verbose=False,
                       extra_cuda_cflags=["-O3", "-fmad=false", "-std=c++17",
                                          f"-gencode=arch=compute_{major}{minor},"
                                          f"code=sm_{major}{minor}"])
    return _EXT
