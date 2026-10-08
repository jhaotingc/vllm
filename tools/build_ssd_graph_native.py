# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the CPU helper for the experimental SSD prefill CUDA graph cache."""

import argparse
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load


def build_ssd_graph_native(build_directory: Path) -> str:
    if CUDA_HOME is None:
        raise RuntimeError("Building the SSD graph helper requires a CUDA toolkit")
    cuda = Path(CUDA_HOME)
    source = Path(__file__).resolve().parents[1] / (
        "vllm/model_executor/layers/mamba/ops/ssd_graph_native.cpp"
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    module = load(
        name="nano35_ssd_graph_native_v1",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[str(cuda / "include")],
        extra_ldflags=[f"-L{cuda / 'lib64/stubs'}", "-lcuda"],
        with_cuda=False,
        build_directory=str(build_directory),
        verbose=True,
    )
    return module.__file__


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_ssd_graph_native(args.build_dir.resolve()))
